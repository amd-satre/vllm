# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP6 (activations) × MXFP4 (weights) Linear method.

Backed by the FlyDSL preshuffle GEMM in ``src/flydsl/preshuffle_gemm_a6w4.py``
from the ``mxfp6_experiments`` repo
(https://gitenterprise.xilinx.com/satre/mxfp6_experiments).

On-disk weight format
---------------------
Identical to compressed-tensors ``mxfp4-pack-quantized``:

  * ``weight_packed``: ``uint8 [out, in / 2]``   — two FP4 codes per byte
  * ``weight_scale`` : ``uint8 [out, in / 32]``  — E8M0 per-32-block scales

Activations are quantized per-token to MXFP6 (E2M3) with matching E8M0/32
scales inside ``apply``.

Kernel constraints
------------------
The FlyDSL kernel requires ``M >= 32 && M % 32 == 0``, ``N % 128 == 0``,
``K % 256 == 0``. Layers that violate N or K alignment are permanently
bf16-dequantized at load. Calls with ``M < 32 or M % 32 != 0`` fall back
to bf16 for that call only.

Environment variables
---------------------
``MXFP6_EXPERIMENTS_FLYDSL_ROOT``   default ``/workspaces/FlyDSL``
``MXFP6_EXPERIMENTS_ROOT``          default ``/home/satre/mxfp6_experiments``
``VLLM_MXFP6_A4_DISABLE_KERNEL=1``  force bf16 dequant for every layer
                                    (useful for accuracy A/B vs the kernel).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)

# ── External checkout locations ────────────────────────────────────────────
_FLYDSL_ROOT = os.environ.get("MXFP6_EXPERIMENTS_FLYDSL_ROOT", "/workspaces/FlyDSL")
_MXFP6_REPO = os.environ.get("MXFP6_EXPERIMENTS_ROOT", "/home/satre/mxfp6_experiments")
_DISABLE_KERNEL = os.environ.get("VLLM_MXFP6_A4_DISABLE_KERNEL", "") == "1"

# Module-level set of prefixes that have already logged a fallback warning,
# so the log stays quiet across calls.
_fallback_warned: set[str] = set()


# ────────────────────────────────────────────────────────────────────────────
# Lazy loaders for the external deps
# ────────────────────────────────────────────────────────────────────────────


_flydsl_fp4_utils: Any = None  # FlyDSL's tests/kernels/utils/fp4_utils.py
_flydsl_compiler: Any = None  # pip-installed `flydsl.compiler`
_a6w4_compile_fn: Any = (
    None  # src.flydsl.preshuffle_gemm_a6w4.compile_preshuffle_gemm_a6w4
)


def _load_flydsl_fp4_utils() -> Any:
    """Load FlyDSL's ``tests/kernels/utils/fp4_utils.py`` via importlib.

    We cannot ``import tests.kernels.utils.fp4_utils`` directly because
    vLLM's own ``tests/`` package shadows the namespace under pytest /
    when ``/workspaces/vllm`` is on ``sys.path``.
    """
    global _flydsl_fp4_utils
    if _flydsl_fp4_utils is not None:
        return _flydsl_fp4_utils
    path = os.path.join(_FLYDSL_ROOT, "tests", "kernels", "utils", "fp4_utils.py")
    if not os.path.isfile(path):
        raise ImportError(
            f"FlyDSL fp4_utils not found at {path}. Set "
            "MXFP6_EXPERIMENTS_FLYDSL_ROOT to the FlyDSL checkout root."
        )
    spec = importlib.util.spec_from_file_location("_flydsl_fp4_utils", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_flydsl_fp4_utils"] = mod
    spec.loader.exec_module(mod)
    _flydsl_fp4_utils = mod
    return mod


def _load_a6w4_compile_fn() -> Any:
    """Load ``compile_preshuffle_gemm_a6w4`` from the mxfp6_experiments src/.

    Adds ``/workspaces/FlyDSL`` (for ``kernels.*``) and
    ``/home/satre/mxfp6_experiments`` (for ``src.flydsl.*``) to ``sys.path``.
    Neither path collides with vLLM's own package namespace (``kernels``
    and ``src`` are absent from vLLM).
    """
    global _a6w4_compile_fn, _flydsl_compiler
    if _a6w4_compile_fn is not None:
        return _a6w4_compile_fn
    for p in (_FLYDSL_ROOT, _MXFP6_REPO):
        if p and p not in sys.path and os.path.isdir(p):
            sys.path.insert(0, p)
    _flydsl_compiler = importlib.import_module("flydsl.compiler")
    mod = importlib.import_module("src.flydsl.preshuffle_gemm_a6w4")
    _a6w4_compile_fn = mod.compile_preshuffle_gemm_a6w4
    return _a6w4_compile_fn


def _flyc():  # noqa: ANN202
    if _flydsl_compiler is None:
        _load_a6w4_compile_fn()
    return _flydsl_compiler


# ────────────────────────────────────────────────────────────────────────────
# In-line OCP MX quantizers (mirrors mxfp6_experiments scripts/flydsl/bench/utils.py)
# ────────────────────────────────────────────────────────────────────────────

_MX_BLOCK = 32
_FP6_E2M3_MAX = 7.5  # E2M3 max normal: 2^(2^1) * (1 + 7/8) = 4 * 1.875


def _mx_e8m0_scale(x_f32: torch.Tensor, *, max_normal: float) -> torch.Tensor:
    """Per-32-block E8M0 scale byte for ``x_f32``.

    Returns ``uint8`` of shape ``x_f32.shape[:-1] + (K // 32,)``.
    """
    fp4u = _load_flydsl_fp4_utils()
    blocks = x_f32.unflatten(-1, (-1, _MX_BLOCK))
    amax = blocks.abs().amax(dim=-1)
    scale_f32 = (amax / max_normal).clamp_(min=2**-126)
    return fp4u.f32_to_e8m0(scale_f32)


def _per_token_mxfp6_e2m3(x_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x_f32 [M, K]`` to MXFP6 E2M3 with per-32-block E8M0 scales.

    Returns:
        ``x_q_unpacked``: ``uint8 [M, K]`` (low 6 bits valid)
        ``scales``      : ``uint8 [M, K // 32]`` (E8M0)
    """
    fp4u = _load_flydsl_fp4_utils()
    scales = _mx_e8m0_scale(x_f32, max_normal=_FP6_E2M3_MAX)
    scale_f32 = fp4u.e8m0_to_f32(scales).repeat_interleave(_MX_BLOCK, dim=-1)
    x_scaled = (x_f32 / scale_f32).contiguous()
    x_q = fp4u._f32_to_floatx_unpacked(x_scaled, 2, 3)  # E2M3
    return x_q, scales


def _pack_fp6_e2m3(x_unpacked: torch.Tensor) -> torch.Tensor:
    """Pack ``uint8 [..., 4G]`` (low 6 bits valid) into ``uint8 [..., 3G]``.

    Little-endian layout per 4-element group:
        byte0 = (elem1[1:0] << 6) | elem0
        byte1 = (elem2[3:0] << 4) | (elem1 >> 2)
        byte2 = (elem3      << 2) | (elem2 >> 4)
    """
    g = x_unpacked.unflatten(-1, (-1, 4)).to(torch.int32) & 0x3F
    e0, e1, e2, e3 = g.unbind(dim=-1)
    b0 = ((e1 & 0x03) << 6) | e0
    b1 = ((e2 & 0x0F) << 4) | (e1 >> 2)
    b2 = (e3 << 2) | (e2 >> 4)
    packed = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8)
    out_shape = x_unpacked.shape[:-1] + (x_unpacked.shape[-1] // 4 * 3,)
    return packed.reshape(out_shape).contiguous()


# ────────────────────────────────────────────────────────────────────────────
# Fused MXFP6 E2M3 activation quant + pack — Triton kernel
# ────────────────────────────────────────────────────────────────────────────
# Replaces the ~20-op torch chain (_per_token_mxfp6_e2m3 ∘ _pack_fp6_e2m3)
# with a single launch. Bit-identical to the reference on finite inputs.
#
# Inputs : x [M, K] (bf16 or fp16 or fp32)
# Outputs: packed [M, K * 3 // 4]  uint8    (3 bytes per 4 elements)
#          scales [M, K // 32]     uint8    (E8M0 per 32-element block)
#
# Constraints: K % 32 == 0 (MX block size). BLOCK_K (compile-time) must
# divide K and be a multiple of 32.

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _mxfp6_e2m3_quant_pack_kernel(
        x_ptr,
        packed_ptr,
        scale_ptr,
        M,
        K,
        stride_xm,
        stride_xk,
        stride_pm,
        stride_pk,
        stride_sm,
        stride_sk,
        BLOCK_K: tl.constexpr,
    ):
        # Grid: (M, K // BLOCK_K)
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        # E2M3 constants -------------------------------------------------------
        # Matches FlyDSL/torchao _f32_to_floatx_unpacked(x, ebits=2, mbits=3)
        EBITS: tl.constexpr = 2
        MBITS: tl.constexpr = 3
        EBITS_F32: tl.constexpr = 8
        MBITS_F32: tl.constexpr = 23
        F32_EXP_BIAS: tl.constexpr = 127
        EXP_BIAS: tl.constexpr = 1  # 2^(EBITS-1) - 1
        MAX_INT: tl.constexpr = 0x1F  # (1 << (EBITS+MBITS)) - 1
        SIGN_MASK: tl.constexpr = 0x20  # 1 << (EBITS+MBITS)
        MAGIC_ADDER: tl.constexpr = (1 << (MBITS_F32 - MBITS - 1)) - 1  # 524287
        # max normal = 2^(2^EBITS - 1 - EXP_BIAS) * (1 + (2^MBITS - 1)/2^MBITS)
        MAX_NORMAL: tl.constexpr = 7.5
        MIN_NORMAL: tl.constexpr = 1.0  # 2^(1 - EXP_BIAS)
        # denorm magic float =
        #   2^(F32_EXP_BIAS - EXP_BIAS + MBITS_F32 - MBITS + 1 - F32_EXP_BIAS)
        # = 2^(147 - 127) = 2^20 = 1048576.0
        DENORM_MASK_INT: tl.constexpr = 147 << 23  # 0x49800000
        DENORM_MASK_FLOAT: tl.constexpr = 1048576.0
        # Right-shift to align f32 sign bit (bit 31) with the fp6 sign bit
        # (bit 5): >> (23 + 8 - 3 - 2) = >> 26. (Reference uses >> 25 because it
        # masks afterwards with SIGN_MASK=0x20; >> 26 lands the sign exactly on
        # bit 5.)
        SIGN_SHIFT: tl.constexpr = MBITS_F32 + EBITS_F32 - MBITS - EBITS  # 26

        # ── Per-block geometry ───────────────────────────────────────────────
        N_BLOCKS: tl.constexpr = BLOCK_K // 32
        N_GROUPS: tl.constexpr = BLOCK_K // 4

        # ── Load BLOCK_K elements of row pid_m, starting at pid_k * BLOCK_K ──
        k_base = pid_k * BLOCK_K
        k_off = k_base + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + pid_m * stride_xm + k_off * stride_xk,
        )
        x_f32 = x.to(tl.float32)

        # ── Per-32-block amax → E8M0 scale (bit-exact f32_to_e8m0) ───────────
        x_blocks = tl.reshape(x_f32, (N_BLOCKS, 32))
        amax = tl.max(tl.abs(x_blocks), axis=1)  # [N_BLOCKS]
        scale_f32 = amax * (1.0 / MAX_NORMAL)
        # clamp to 2^-126 (smallest normal f32) so the e8m0 conversion below
        # doesn't observe a subnormal-encoded scale
        scale_f32 = tl.maximum(scale_f32, 1.1754943508222875e-38)

        s_u32 = scale_f32.to(tl.uint32, bitcast=True)
        exponent = ((s_u32 >> 23) & 0xFF).to(tl.int32)
        nan_case = exponent == 0xFF
        round_case = ((s_u32 & 0x400000) != 0) & (
            ((s_u32 & 0x200000) != 0) | ((s_u32 & 0x1FFFFF) != 0) | (exponent > 0)
        )
        exp_rounded = exponent + round_case.to(tl.int32)
        exp_final = tl.where(nan_case, 0xFF, exp_rounded)
        e8m0 = exp_final.to(tl.uint8)  # [N_BLOCKS]

        # ── Rebuild scale as f32 from e8m0: scale_back = 2^(e8m0 - 127) ──────
        scale_back_u32 = e8m0.to(tl.uint32) << 23
        scale_back = scale_back_u32.to(tl.float32, bitcast=True)  # [N_BLOCKS]

        # Broadcast scale to BLOCK_K and divide
        scale_full = tl.reshape(
            scale_back[:, None] * tl.full((1, 32), 1.0, dtype=tl.float32),
            (BLOCK_K,),
        )
        x_scaled = x_f32 / scale_full

        # ── f32 → fp6_e2m3 unpacked (bit-exact _f32_to_floatx_unpacked) ──────
        # NOTE: use uint32 for sign extraction since 0x80000000 > INT32_MAX.
        x_u32 = x_scaled.to(tl.uint32, bitcast=True)
        sign = x_u32 & 0x80000000
        x_abs_u32 = x_u32 ^ sign
        x_abs = x_abs_u32.to(tl.float32, bitcast=True)

        sat_mask = x_abs >= MAX_NORMAL
        denorm_mask = (x_abs < MIN_NORMAL) & (~sat_mask)
        norm_mask = ~(sat_mask | denorm_mask)

        # Branch 2: denormal
        denorm_x_f = x_abs + DENORM_MASK_FLOAT
        denorm_x_i = denorm_x_f.to(tl.int32, bitcast=True) - DENORM_MASK_INT
        denorm_x_u8 = denorm_x_i.to(tl.uint8)

        # Branch 3: normal
        normal_xi = x_abs.to(tl.int32, bitcast=True)
        mant_odd = (normal_xi >> (MBITS_F32 - MBITS)) & 1
        val_to_add = ((EXP_BIAS - F32_EXP_BIAS) << MBITS_F32) + MAGIC_ADDER
        normal_xi = normal_xi + val_to_add + mant_odd
        normal_xi = normal_xi >> (MBITS_F32 - MBITS)
        normal_x_u8 = normal_xi.to(tl.uint8)

        # Combine (default = saturate to MAX_INT)
        sat_u8 = tl.full((BLOCK_K,), MAX_INT, dtype=tl.uint8)
        out = tl.where(
            denorm_mask,
            denorm_x_u8,
            tl.where(norm_mask, normal_x_u8, sat_u8),
        )

        # Add sign bit at position 5
        sign_lp = (sign >> SIGN_SHIFT).to(tl.uint8) & SIGN_MASK
        out = out | sign_lp

        # ── Pack 4 elements → 3 bytes via shift-sum (no overlap, so OR≡sum) ──
        out_g = tl.reshape(out, (N_GROUPS, 4)).to(tl.uint32) & 0x3F
        shifts = tl.arange(0, 4) * 6  # [4]: 0, 6, 12, 18
        packed_u32 = tl.sum(out_g << shifts[None, :], axis=1)  # [N_GROUPS]
        b0 = (packed_u32 & 0xFF).to(tl.uint8)
        b1 = ((packed_u32 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_u32 >> 16) & 0xFF).to(tl.uint8)

        # ── Store packed bytes: 3 contiguous bytes per group ─────────────────
        group_idx = tl.arange(0, N_GROUPS)
        pack_base = pid_k * (BLOCK_K // 4 * 3)
        group_off = pack_base + group_idx * 3
        tl.store(packed_ptr + pid_m * stride_pm + (group_off + 0) * stride_pk, b0)
        tl.store(packed_ptr + pid_m * stride_pm + (group_off + 1) * stride_pk, b1)
        tl.store(packed_ptr + pid_m * stride_pm + (group_off + 2) * stride_pk, b2)

        # ── Store scales: N_BLOCKS bytes per program ─────────────────────────
        s_off = pid_k * N_BLOCKS + tl.arange(0, N_BLOCKS)
        tl.store(scale_ptr + pid_m * stride_sm + s_off * stride_sk, e8m0)


def per_token_mxfp6_e2m3_packed_triton(
    x: torch.Tensor,
    *,
    block_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused per-token MXFP6 E2M3 quant + pack via one Triton launch.

    Args:
        x: [M, K] (bf16/fp16/fp32). K must be a multiple of ``block_k``,
           which must in turn be a multiple of 32.
        block_k: kernel BLOCK_K (default 128 = 4 MX blocks per program).

    Returns:
        packed: uint8 [M, K * 3 // 4]
        scales: uint8 [M, K // 32]   (interpret as E8M0)
    """
    assert _HAS_TRITON, "Triton not available"
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    assert x.is_contiguous(), "x must be contiguous"
    M, K = x.shape
    assert K % block_k == 0, f"K={K} must be divisible by block_k={block_k}"
    assert block_k % 32 == 0, f"block_k={block_k} must be a multiple of 32"

    packed = torch.empty((M, K * 3 // 4), dtype=torch.uint8, device=x.device)
    scales = torch.empty((M, K // 32), dtype=torch.uint8, device=x.device)

    grid = (M, K // block_k)
    _mxfp6_e2m3_quant_pack_kernel[grid](
        x,
        packed,
        scales,
        M,
        K,
        x.stride(0),
        x.stride(1),
        packed.stride(0),
        packed.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_K=block_k,
    )
    # Match reference dtype: scales bytes are E8M0 codepoints.
    if hasattr(torch, "float8_e8m0fnu"):
        scales = scales.view(torch.float8_e8m0fnu)
    return packed, scales


# ────────────────────────────────────────────────────────────────────────────
# Kernel tile picker (mirrors mxfp6_experiments _pick_a6w4_config defaults)
# ────────────────────────────────────────────────────────────────────────────


def _pick_kernel_config(
    M: int, N: int, K: int
) -> tuple[int, int, int, int, bool, int | None]:
    """Return (tile_m, tile_n, tile_k, lds_stage, use_async_copy, waves_per_eu).

    The default tile-picking logic mirrors
    ``scripts/flydsl/bench/benchmark.py::_pick_a6w4_tiles``. Per-shape tuned
    overrides from ``A6W4_TUNED_CONFIGS`` are loaded lazily if the
    mxfp6_experiments repo is on ``sys.path``.
    """
    if M >= 128 and M % 128 == 0:
        tile_m = 128
    elif M >= 64 and M % 64 == 0:
        tile_m = 64
    else:
        tile_m = 32
    tile_n = 256 if N % 256 == 0 else 128
    tile_k = 256 if K % 256 == 0 else 128
    cfg: dict[str, Any] = {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "lds_stage": 2,
        "use_async_copy": False,
        "waves_per_eu": None,
    }
    tuned = _load_tuned_config_dict()
    if tuned is not None:
        cfg.update(tuned.get((M, N, K), {}))
    return (
        cfg["tile_m"],
        cfg["tile_n"],
        cfg["tile_k"],
        cfg["lds_stage"],
        cfg["use_async_copy"],
        cfg["waves_per_eu"],
    )


_tuned_cfg_cache: dict | None = None
_tuned_cfg_attempted = False


def _load_tuned_config_dict() -> dict | None:
    """Lazy-load ``A6W4_TUNED_CONFIGS`` from mxfp6_experiments by direct file
    read — avoids triggering the rest of benchmark.py's imports."""
    global _tuned_cfg_cache, _tuned_cfg_attempted
    if _tuned_cfg_attempted:
        return _tuned_cfg_cache
    _tuned_cfg_attempted = True
    path = os.path.join(_MXFP6_REPO, "scripts", "flydsl", "bench", "benchmark.py")
    if not os.path.isfile(path):
        return None
    # Use ast to extract just the dict literal, no execution.
    import ast

    try:
        with open(path) as _f:
            tree = ast.parse(_f.read(), filename=path)
    except Exception:  # noqa: BLE001
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "A6W4_TUNED_CONFIGS"
            for t in node.targets
        ):
            try:
                _tuned_cfg_cache = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
            break
    return _tuned_cfg_cache


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────


def _kernel_alignment_ok(N: int, K: int) -> bool:
    return N % 128 == 0 and K % 256 == 0


class Mxfp6A4Config(QuantizationConfig):
    """Quantization config for ``mxfp6_a4`` (FlyDSL MXFP6 A × MXFP4 B GEMM)."""

    def __init__(
        self,
        group_size: int = 32,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        if group_size != 32:
            raise ValueError(f"mxfp6_a4 only supports group_size=32 (got {group_size})")
        self.group_size = group_size
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp6_a4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # MI355X / gfx950. The capability number is a no-op on AMD; vLLM only
        # uses it to gate against older NVIDIA GPUs.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Mxfp6A4Config:
        group_size = int(config.get("group_size", 32))
        ignored_layers = config.get("ignored_layers", []) or []
        return cls(group_size=group_size, ignored_layers=ignored_layers)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            is_layer_skipped,
        )

        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return Mxfp6A4LinearMethod(self, prefix=prefix)
        # FusedMoE / Attention — leave to the framework's defaults for now.
        return None

    def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool:
        return True


# ────────────────────────────────────────────────────────────────────────────
# Linear method
# ────────────────────────────────────────────────────────────────────────────


class Mxfp6A4LinearMethod(LinearMethodBase):
    """Per-layer Linear method that dispatches the FlyDSL a6w4 kernel.

    Layer attributes added:
      * ``weight_packed``, ``weight_scale``    — set in ``create_weights``,
        populated by the model's weight loader.
      * ``_mxfp6_a4_can_kernel``               — True if the per-rank (N, K)
        satisfy the kernel alignment constraints. False means the layer goes
        through the bf16 dequant path for every forward call.
      * ``_weight_b_shuf``, ``_weight_sb_shuf`` — preshuffled MXFP4 weight and
        E8M0 scale tensors (only for kernel-eligible layers).
      * ``_mxfp6_a4_launchers``                — dict
        ``(M, N, K, cfg) -> compiled launcher``.
      * ``_mxfp6_a4_w_bf16``                   — lazily-built dequantized
        weight; only materialized when a forward call lands on the bf16
        fallback path.
    """

    def __init__(self, config: Mxfp6A4Config, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.group_size = config.group_size

    # ── Weight registration ────────────────────────────────────────────────

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_loader: Callable = extra_weight_attrs["weight_loader"]

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        # ── Packed MXFP4 weight (matches compressed-tensors mxfp4-pack layout)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # ── Per-32-block E8M0 scales
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        layer._mxfp6_a4_can_kernel = (
            _kernel_alignment_ok(output_size_per_partition, input_size_per_partition)
            and not _DISABLE_KERNEL
        )
        if not layer._mxfp6_a4_can_kernel:
            reason = (
                "VLLM_MXFP6_A4_DISABLE_KERNEL=1"
                if _DISABLE_KERNEL
                else f"N={output_size_per_partition} or "
                f"K={input_size_per_partition} violates "
                "(N%128, K%256) alignment"
            )
            self._warn_fallback(
                f"layer '{self.prefix}' will always use bf16 dequant: {reason}"
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not layer._mxfp6_a4_can_kernel:
            return  # bf16 path is built lazily on first forward call

        try:
            fp4u = _load_flydsl_fp4_utils()
        except ImportError as exc:
            self._warn_fallback(
                f"layer '{self.prefix}': FlyDSL fp4_utils not importable "
                f"({exc}); using bf16 dequant fallback."
            )
            layer._mxfp6_a4_can_kernel = False
            return

        with torch.no_grad():
            b_packed = layer.weight_packed.data.contiguous()
            b_scales = layer.weight_scale.data.contiguous()
            b_shuf = fp4u.shuffle_weight_w4(b_packed, 16, False, False)
            sb_shuf = fp4u.shuffle_scale_w4(b_scales, 1, False)

        layer.register_parameter(
            "_weight_b_shuf", Parameter(b_shuf, requires_grad=False)
        )
        layer.register_parameter(
            "_weight_sb_shuf", Parameter(sb_shuf, requires_grad=False)
        )
        layer._mxfp6_a4_launchers = {}

    # ── Forward ────────────────────────────────────────────────────────────

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = layer.output_size_per_partition
        K = layer.input_size_per_partition
        x2d = x.reshape(-1, K)
        M_real = x2d.shape[0]

        # Layer-level disable: kernel alignment failed at load OR forced off.
        if not layer._mxfp6_a4_can_kernel:
            return self._bf16_fallback(layer, x, bias)

        # Per-call disable: small / non-32 M (e.g. decode). Padding M=1 → 32
        # would do 32× the work; bf16 dequant is faster.
        if M_real < 32 or M_real % 32 != 0:
            return self._bf16_fallback(layer, x, bias)

        try:
            y2d = self._kernel_apply(layer, x2d, M_real, N, K)
        except Exception as exc:  # noqa: BLE001  — kernel issues fail soft
            self._warn_fallback(
                f"layer '{self.prefix}': kernel call raised {type(exc).__name__}"
                f": {exc}; using bf16 dequant fallback."
            )
            return self._bf16_fallback(layer, x, bias)

        if bias is not None:
            y2d = y2d + bias
        return y2d.reshape(*x.shape[:-1], N)

    # ── Internals ──────────────────────────────────────────────────────────

    def _kernel_apply(
        self,
        layer: torch.nn.Module,
        x2d: torch.Tensor,
        M: int,
        N: int,
        K: int,
    ) -> torch.Tensor:
        compile_fn = _load_a6w4_compile_fn()
        fp4u = _load_flydsl_fp4_utils()
        flyc = _flyc()

        device = x2d.device

        # ── Per-token activation quant: bf16 → MXFP6 (E2M3) + E8M0 scales
        x_f32 = x2d.float().contiguous()
        a_unpacked, a_scales = _per_token_mxfp6_e2m3(x_f32)  # (M,K) low-6, (M,K/32)
        a_packed24 = _pack_fp6_e2m3(a_unpacked)  # (M, K*3/4)
        nblk = K // 32
        a_kernel = torch.zeros((M, K), device=device, dtype=torch.uint8)
        a_kernel.view(M, nblk, 32)[:, :, :24] = a_packed24.view(M, nblk, 24)
        sa_shuf = fp4u.shuffle_scale_w4(a_scales, 1, False)
        # FlyDSL's DLTensorAdaptor doesn't handle DLPack code 14
        # (float8_e8m0fnu); view as uint8 (matches the _to_bytes() trick in
        # FlyDSL's own tests/kernels/test_preshuffle_gemm.py).
        if sa_shuf.dtype != torch.uint8:
            sa_shuf = sa_shuf.view(torch.uint8)

        cfg = _pick_kernel_config(M, N, K)
        key = (M, N, K, cfg)
        compiled = layer._mxfp6_a4_launchers.get(key)
        if compiled is None:
            tile_m, tile_n, tile_k, lds_stage, use_async, waves_per_eu = cfg
            launch_fn = compile_fn(
                M=M,
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
            c_seed = torch.zeros((M, N), device=device, dtype=torch.bfloat16)
            dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)
            seed_args = (
                c_seed.view(-1),
                a_kernel.view(-1),
                layer._weight_b_shuf.data.view(-1),
                sa_shuf.view(-1),
                layer._weight_sb_shuf.data.view(-1),
                dummy_bias,
                M,
                N,
                torch.cuda.current_stream(),
            )
            compiled = flyc.compile(launch_fn, *seed_args)
            layer._mxfp6_a4_launchers[key] = compiled

        c_bf16 = torch.zeros((M, N), device=device, dtype=torch.bfloat16)
        dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)
        compiled(
            c_bf16.view(-1),
            a_kernel.view(-1),
            layer._weight_b_shuf.data.view(-1),
            sa_shuf.view(-1),
            layer._weight_sb_shuf.data.view(-1),
            dummy_bias,
            M,
            N,
            torch.cuda.current_stream(),
        )
        return c_bf16

    def _bf16_fallback(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        w_bf16 = self._dequant_weight_to_bf16(layer)
        return torch.nn.functional.linear(x, w_bf16, bias)

    @staticmethod
    def _dequant_weight_to_bf16(layer: torch.nn.Module) -> torch.Tensor:
        cached = getattr(layer, "_mxfp6_a4_w_bf16", None)
        if cached is not None:
            return cached
        fp4u = _load_flydsl_fp4_utils()
        b_f32 = fp4u.mxfp4_to_f32(layer.weight_packed.data)
        s_f32 = fp4u.e8m0_to_f32(layer.weight_scale.data).repeat_interleave(
            _MX_BLOCK, dim=-1
        )
        w_bf16 = (b_f32 * s_f32[..., : b_f32.shape[-1]]).to(torch.bfloat16)
        layer._mxfp6_a4_w_bf16 = w_bf16
        return w_bf16

    def _warn_fallback(self, msg: str) -> None:
        if self.prefix in _fallback_warned:
            return
        _fallback_warned.add(self.prefix)
        logger.warning_once(msg, scope="local")
