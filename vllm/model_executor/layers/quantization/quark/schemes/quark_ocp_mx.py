# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from fractions import Fraction
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    dequant_mxfp6,
    quant_dequant_mxfp6,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
    OCP_MX_Scheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)


# ── FlyDSL a6w4 dispatch (gfx950, opt-in via env) ──────────────────────────
# Reuses the FlyDSL plumbing in vllm.model_executor.layers.quantization.mxfp6_a4
# (loaders for fp4_utils + compile_preshuffle_gemm_a6w4, MXFP6 E2M3 activation
# quantizer, tile-config picker). The Quark loader provides packed-uint8 MXFP4
# weights + uint8 E8M0 scales; we add the FlyDSL-specific preshuffle in
# process_weights_after_loading and call the kernel in apply_weights.
#
# Env: VLLM_MX_USE_FLYDSL=1 enables FlyDSL for the (mxfp4, mxfp6_e2m3) combo.
# Default off; the user opts in per benchmark run.
_VLLM_MX_USE_FLYDSL = os.environ.get("VLLM_MX_USE_FLYDSL", "") == "1"

_mxfp6_a4_helpers: Any = None
try:
    from vllm.model_executor.layers.quantization import mxfp6_a4 as _mxfp6_a4_helpers

    _FLYDSL_HELPERS_AVAILABLE = True
except Exception:  # noqa: BLE001
    _FLYDSL_HELPERS_AVAILABLE = False


# Module-level launcher cache, keyed by (M_pad, N, K, cfg). Launchers
# specialize on shape + tile config; pointers are passed per call.
_flydsl_launcher_cache: dict[tuple, Any] = {}


def _flydsl_dispatch_available(input_dtype: str | None, weight_dtype: str) -> bool:
    """Decide whether to route the (input, weight) combo through FlyDSL."""
    if not _VLLM_MX_USE_FLYDSL:
        return False
    if not _FLYDSL_HELPERS_AVAILABLE:
        return False
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            return False
    except Exception:  # noqa: BLE001
        return False
    # Today: only the (mxfp4 weight, mxfp6_e2m3 act) combo. (mxfp4, mxfp4) is
    # left on AITER's tuned ASM path, which is already fast on gfx950.
    return weight_dtype == "mxfp4" and input_dtype == "mxfp6_e2m3"


def _flydsl_a6w4_apply(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """End-to-end MXFP6×MXFP4 GEMM via FlyDSL.

    x:            bf16 [M, K]
    weight:       uint8 [N, K // 2]   (MXFP4 packed, pre-shuffled in PWAL)
    weight_scale: uint8 [N, K // 32]  (E8M0, pre-shuffled in PWAL)

    Returns: bf16 [M, N]
    """
    assert _FLYDSL_HELPERS_AVAILABLE, "FlyDSL helpers not importable"
    helpers = _mxfp6_a4_helpers

    M_real, K = x.shape
    N = weight.shape[0]
    device = x.device

    # Pad M up to a multiple of 32 (FlyDSL constraint: M >= 32, M % 32 == 0).
    # M=1 (decode) → pad to 32; we waste 32× the GEMM work on the activation
    # side but avoid a separate bf16 dequant kernel. For benchmarking we want
    # the FlyDSL path on every call.
    M_pad = max(32, ((M_real + 31) // 32) * 32)
    if M_pad != M_real:
        x_pad = torch.zeros((M_pad, K), device=device, dtype=x.dtype)
        x_pad[:M_real].copy_(x)
    else:
        x_pad = x.contiguous()

    fp4u = helpers._load_flydsl_fp4_utils()
    compile_fn = helpers._load_a6w4_compile_fn()
    flyc = helpers._flyc()

    # ── Per-token activation quant: bf16 → MXFP6 E2M3 + E8M0 scales ──────
    x_f32 = x_pad.float().contiguous()
    a_unpacked, a_scales = helpers._per_token_mxfp6_e2m3(x_f32)  # (M,K) low-6, (M,K/32)
    a_packed24 = helpers._pack_fp6_e2m3(a_unpacked)  # (M, K*3/4)
    nblk = K // 32
    a_kernel = torch.zeros((M_pad, K), device=device, dtype=torch.uint8)
    a_kernel.view(M_pad, nblk, 32)[:, :, :24] = a_packed24.view(M_pad, nblk, 24)
    sa_shuf = fp4u.shuffle_scale_w4(a_scales, 1, False)
    # FlyDSL's DLTensorAdaptor doesn't handle DLPack code 14
    # (float8_e8m0fnu); view as uint8 (matches the _to_bytes() trick in
    # FlyDSL's own tests/kernels/test_preshuffle_gemm.py).
    if sa_shuf.dtype != torch.uint8:
        sa_shuf = sa_shuf.view(torch.uint8)

    cfg = helpers._pick_kernel_config(M_pad, N, K)
    key = (M_pad, N, K, cfg)
    compiled = _flydsl_launcher_cache.get(key)
    if compiled is None:
        tile_m, tile_n, tile_k, lds_stage, use_async, waves_per_eu = cfg
        launch_fn = compile_fn(
            M=M_pad,
            N=N,
            K=K,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            out_dtype="bf16",
            lds_stage=lds_stage,
            use_async_copy=use_async,
            waves_per_eu=waves_per_eu,
        )
        c_seed = torch.zeros((M_pad, N), device=device, dtype=torch.bfloat16)
        dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)
        seed_args = (
            c_seed.view(-1),
            a_kernel.view(-1),
            weight.view(-1),
            sa_shuf.view(-1),
            weight_scale.view(-1),
            dummy_bias,
            M_pad,
            N,
            torch.cuda.current_stream(),
        )
        compiled = flyc.compile(launch_fn, *seed_args)
        _flydsl_launcher_cache[key] = compiled

    c_bf16 = torch.zeros((M_pad, N), device=device, dtype=torch.bfloat16)
    dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)
    compiled(
        c_bf16.view(-1),
        a_kernel.view(-1),
        weight.view(-1),
        sa_shuf.view(-1),
        weight_scale.view(-1),
        dummy_bias,
        M_pad,
        N,
        torch.cuda.current_stream(),
    )
    y = c_bf16[:M_real]
    if out_dtype != torch.bfloat16:
        y = y.to(out_dtype)
    return y


try:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.gemm_afp4wfp4 import (
        gemm_afp4wfp4,
        gemm_afp4wfp4_preshuffled_weight_scales,
    )
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.utils.torch_utils import direct_register_custom_op

    if rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled():
        from aiter import gemm_a4w4, per_1x32_f4_quant_hip

    def gemm_with_dynamic_quant(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: torch.dtype | None = torch.bfloat16,
        x_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        M = x.shape[0]
        N = weight.shape[0]
        K = weight.shape[1]
        if rocm_use_aiter_fp4_asm_gemm:
            if M <= 64 and rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(N, K):
                if x_scales is None:
                    # use hip quant kernel for performance
                    if M >= 32:
                        x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)
                    else:
                        x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=False)
                else:
                    x_q = x
                    x_s = x_scales

                if M >= 32:
                    x_s = x_s.view(torch.uint8).view(x_s.shape[0] // 32, -1)
                else:
                    x_s = x_s[:M, ...].view(torch.uint8)

                y = torch.empty(M, N, device=x_q.device, dtype=out_dtype)
                gemm_afp4wfp4_preshuffled_weight_scales(
                    x_q.view(torch.uint8),
                    weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
                    x_s,
                    weight_scale.view(torch.uint8).view(
                        weight_scale.shape[0] // 32, -1
                    ),
                    out_dtype,
                    y,
                )
            else:
                if x_scales is None:
                    # use hip quant kernel for performance
                    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)
                else:
                    x_q = x
                    x_s = x_scales

                # 32 alignment is enough for dim0 padding of output for
                # gemm_a4w4 kernel
                y = torch.empty(
                    (M + 31) // 32 * 32,
                    weight.shape[0],
                    device=x_q.device,
                    dtype=out_dtype,
                )

                gemm_a4w4(
                    x_q,
                    weight.view(x_q.dtype),
                    x_s,
                    weight_scale.view(x_s.dtype),
                    y,
                    bpreshuffle=True,
                )
            return y[:M]
        else:
            if x_scales is None:
                x_q, x_s = dynamic_mxfp4_quant(x)
            else:
                x_q = x
                x_s = x_scales
            y = torch.empty(
                x_q.shape[0], weight.shape[0], device=x_q.device, dtype=out_dtype
            )

            gemm_afp4wfp4(x_q, weight, x_s, weight_scale.T, out_dtype, y)
            return y

    def gemm_with_dynamic_quant_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scales: torch.Tensor = None,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: torch.dtype | None = torch.bfloat16,
    ) -> torch.Tensor:
        return torch.empty(
            (*x.shape[:-1], weight.shape[0]), dtype=out_dtype, device=x.device
        )

    direct_register_custom_op(
        op_name="gemm_with_dynamic_quant",
        op_func=gemm_with_dynamic_quant,
        mutates_args=[],
        fake_impl=gemm_with_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )
except (ImportError, AttributeError, RuntimeError):
    if current_platform.is_rocm():
        logger.warning(
            "AITER is not found or QuarkOCP_MX is not supported on the current "
            "platform. QuarkOCP_MX quantization will not be available."
        )
    dynamic_mxfp4_quant = gemm_afp4wfp4 = None


class QuarkOCP_MX(QuarkScheme):
    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any] | None,
        dynamic_mxfp4_quant: bool = False,
        emulation_dequantize_weights: bool = False,
    ):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.dynamic_mxfp4_quant = dynamic_mxfp4_quant
        self.weight_dtype = weight_quant_spec["dtype"].replace("fp", "mxfp")
        self.input_dtype: str | None = None
        if input_quant_spec is not None:
            input_quant = input_quant_spec["dtype"]
            if input_quant == "fp8_e4m3":
                self.input_dtype = "fp8"
            else:
                self.input_dtype = input_quant.replace("fp", "mxfp")

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype
        )

        if self.weight_dtype == "mxfp4":
            self.packed_factor: int | Fraction = 2
            self.dequant_func = dequant_mxfp4
        else:
            self.packed_factor = Fraction(numerator=8, denominator=6)
            self.dequant_func = partial(
                dequant_mxfp6, quant_dtype=self.weight_dtype.replace("mx", "")
            )

        if self.input_dtype is None:
            self.quant_dequant_func: Callable[[torch.Tensor], torch.Tensor] = (
                lambda x: x
            )  # no input Q/DQ for weight-only
        elif self.input_dtype == "mxfp4":
            self.quant_dequant_func = quant_dequant_mxfp4
        else:
            self.quant_dequant_func = partial(
                quant_dequant_mxfp6, quant_dtype=self.input_dtype.replace("mx", "")
            )

        if input_quant_spec is None:
            self.static_input_scales = False
        else:
            self.static_input_scales = not input_quant_spec.get("is_dynamic")

        if self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX with static input scales is currently not "
                "implemented. Please open an issue."
            )

        # TODO: integrate (or test) mixed-precision kernel.
        self.emulate = not current_platform.supports_mx() or (
            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"
        )

        # FlyDSL a6w4 (MXFP4 weights × MXFP6_E2M3 activations) on gfx950:
        # if the user has VLLM_MX_USE_FLYDSL=1 and the dtype combo matches,
        # override the emulate fall-back with a real kernel dispatch.
        self._use_flydsl = _flydsl_dispatch_available(
            self.input_dtype, self.weight_dtype
        )
        if self._use_flydsl:
            self.emulate = False
            logger.info_once(
                "QuarkOCP_MX: routing %s linears through FlyDSL a6w4 kernel.",
                self.ocp_mx_scheme.value if self.ocp_mx_scheme else "<unknown>",
                scope="local",
            )

        self.emulation_dequantize_weights = emulation_dequantize_weights
        if self.emulation_dequantize_weights:
            logger.info_once(
                "QuarkOCP_MX simulated dense linear: "
                "dequantizing weights ahead of time."
            )

        self.rocm_use_aiter_fp4_asm_gemm = (
            rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()
        )

        if (
            not self.emulate
            and not self._use_flydsl
            and (dynamic_mxfp4_quant is None or gemm_afp4wfp4 is None)
        ):
            # Currently need these kernels if not emulating
            raise NotImplementedError(
                f"{self.__class__.__name__} requires AITER to be installed "
                "for non-emulation mode! Please refer to "
                "https://github.com/ROCm/aiter for installation details."
            )

        if not current_platform.supports_mx():
            logger.warning_once(
                "The current platform does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

        if (
            current_platform.supports_mx()
            and not self._use_flydsl
            and (self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4")
        ):
            logger.warning_once(
                "The current platform supports native MXFP4/MXFP6 "
                f"computation, but kernels for input_dtype={self.input_dtype} "
                f"and weight_dtype={self.weight_dtype} are not yet integrated "
                "in vLLM. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype == "mxfp4":
            assert dim % 2 == 0
            return dim // 2
        elif quant_dtype in {"mxfp6_e3m2", "mxfp6_e2m3"}:
            # FP6 packs 4 * 6 = 24 bits on 3 bytes.
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4
        else:
            raise NotImplementedError(
                "Unsupported quant_dtype in QuarkOCP_MX.get_packed_dim, "
                f"got quant_dtype={quant_dtype}. Something is wrong, please "
                "open an issue."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_dynamic_mxfp4_weights_after_loading(
        self, layer: torch.nn.Module
    ) -> None:
        w_q, w_s = dynamic_mxfp4_quant(layer.weight)
        layer.weight_scale = torch.nn.Parameter(w_s.T.contiguous(), requires_grad=False)
        layer.weight = torch.nn.Parameter(w_q, requires_grad=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)

        if self._use_flydsl:
            fp4u = _mxfp6_a4_helpers._load_flydsl_fp4_utils()
            with torch.no_grad():
                b_shuf = fp4u.shuffle_weight_w4(
                    layer.weight.data.contiguous(), 16, False, False
                )
                sb_shuf = fp4u.shuffle_scale_w4(
                    layer.weight_scale.data.contiguous(), 1, False
                )
            layer.weight = torch.nn.Parameter(b_shuf, requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(sb_shuf, requires_grad=False)
            return

        if self.emulate:
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )

            if self.emulation_dequantize_weights:
                dq_w = self.dequant_func(
                    layer.weight, layer.weight_scale, torch.get_default_dtype()
                )
                layer.weight = torch.nn.Parameter(dq_w, requires_grad=False)
                layer.weight_scale = None
        else:
            if self.dynamic_mxfp4_quant:
                self.process_dynamic_mxfp4_weights_after_loading(layer)
            elif self.rocm_use_aiter_fp4_asm_gemm:
                # shuffle weight scale
                weight_scale_shuffle = layer.weight_scale.data
                sm, sn = weight_scale_shuffle.shape
                weight_scale_shuffle = weight_scale_shuffle.view(
                    sm // 32, 2, 16, sn // 8, 2, 4, 1
                )
                weight_scale_shuffle = weight_scale_shuffle.permute(
                    0, 3, 5, 2, 4, 1, 6
                ).contiguous()
                weight_scale_shuffle = weight_scale_shuffle.view(sm, sn)
                layer.weight_scale = torch.nn.Parameter(
                    weight_scale_shuffle, requires_grad=False
                )

                # shuffle weight
                weight_shuffle = layer.weight.data
                weight_shuffle = shuffle_weight(weight_shuffle, layout=(16, 16))
                layer.weight = torch.nn.Parameter(weight_shuffle, requires_grad=False)
            else:
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale.data.T.contiguous(), requires_grad=False
                )

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        if self.dynamic_mxfp4_quant:
            weight = ModelWeightParameter(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )

            layer.register_parameter("weight", weight)
            set_weight_attrs(weight, kwargs)
        else:
            output_size_per_partition = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes

            # WEIGHT
            weight = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    self.get_packed_dim(input_size_per_partition, self.weight_dtype),
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=self.packed_factor,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)

            # WEIGHT SCALE
            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // OCP_MX_BLOCK_SIZE,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._use_flydsl:
            x2d = x.reshape(-1, x.shape[-1])
            y2d = _flydsl_a6w4_apply(
                x2d, layer.weight, layer.weight_scale, self.out_dtype
            )
            if bias is not None:
                y2d = y2d + bias
            return y2d.reshape(*x.shape[:-1], y2d.shape[-1])

        if self.emulate:
            if not self.emulation_dequantize_weights:
                dq_w = self.dequant_func(layer.weight, layer.weight_scale, x.dtype)
            else:
                dq_w = layer.weight

            qdq_x = self.quant_dequant_func(x)
            return F.linear(qdq_x, dq_w, bias)
        else:
            return torch.ops.vllm.gemm_with_dynamic_quant(
                x,
                layer.weight,
                layer.weight_scale,
                self.rocm_use_aiter_fp4_asm_gemm,
                self.out_dtype,
            )
