# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import utils as fused_moe_utils
from vllm.model_executor.layers.fused_moe.experts import ocp_mx_emulation_moe
from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
    OCP_MXQuantizationEmulationTritonExperts,
    _ocp_mx_emulated_activation_dtype,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    _should_emulate_ocp_mx_moe,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_Scheme,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("ocp_mx_scheme", "expected_activation_dtype"),
    [
        (OCP_MX_Scheme.w_mxfp4, None),
        (OCP_MX_Scheme.w_mxfp6_e3m2, None),
        (OCP_MX_Scheme.w_mxfp6_e2m3, None),
        (OCP_MX_Scheme.w_mxfp4_a_mxfp4, "mxfp4"),
        (OCP_MX_Scheme.w_mxfp4_a_mxfp6_e3m2, "mxfp6_e3m2"),
        (OCP_MX_Scheme.w_mxfp4_a_mxfp6_e2m3, "mxfp6_e2m3"),
        (OCP_MX_Scheme.w_mxfp6_e3m2_a_mxfp6_e3m2, "mxfp6_e3m2"),
        (OCP_MX_Scheme.w_mxfp6_e2m3_a_mxfp6_e2m3, "mxfp6_e2m3"),
        (OCP_MX_Scheme.w_mxfp4_a_fp8, "mxfp8"),
        (OCP_MX_Scheme.w_mxfp6_e3m2_a_fp8, "mxfp8"),
    ],
)
def test_ocp_mx_emulation_activation_dtype(
    ocp_mx_scheme: OCP_MX_Scheme,
    expected_activation_dtype: str | None,
) -> None:
    assert _ocp_mx_emulated_activation_dtype(ocp_mx_scheme) == expected_activation_dtype


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("ocp_mx_scheme", "expected_activation_dtype"),
    [
        (OCP_MX_Scheme.w_mxfp4, None),
        (OCP_MX_Scheme.w_mxfp4_a_mxfp4, "mxfp4"),
    ],
)
def test_ocp_mx_emulation_apply_uses_scheme_activation_dtype(
    monkeypatch: pytest.MonkeyPatch,
    ocp_mx_scheme: OCP_MX_Scheme,
    expected_activation_dtype: str | None,
) -> None:
    class StopAfterQuantize(Exception):
        pass

    experts = object.__new__(OCP_MXQuantizationEmulationTritonExperts)
    experts.ocp_mx_scheme = ocp_mx_scheme
    experts._quant_dtype = _ocp_mx_emulated_activation_dtype(ocp_mx_scheme)
    experts.quant_config = SimpleNamespace(quant_dtype="mxfp4")
    experts.w1_scale_val = torch.empty(1, dtype=torch.uint8)
    experts.w2_scale_val = torch.empty(1, dtype=torch.uint8)

    def fake_dequantize_weights(
        _self: OCP_MXQuantizationEmulationTritonExperts,
        _w: torch.Tensor,
        _w_scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.empty(1, 1, 1, dtype=dtype)

    captured: dict[str, torch.dtype | str | None] = {}

    def fake_quantize_input(
        A: torch.Tensor,
        A_scale: torch.Tensor | None,
        quant_dtype: torch.dtype | str | None,
        per_act_token_quant: bool,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del A, A_scale, per_act_token_quant
        captured["quant_dtype"] = quant_dtype
        raise StopAfterQuantize

    monkeypatch.setattr(
        OCP_MXQuantizationEmulationTritonExperts,
        "_dequantize_weights",
        fake_dequantize_weights,
    )
    monkeypatch.setattr(
        ocp_mx_emulation_moe,
        "moe_kernel_quantize_input",
        fake_quantize_input,
    )

    with pytest.raises(StopAfterQuantize):
        experts.apply(
            output=torch.empty(1, 1),
            hidden_states=torch.empty(1, 1, dtype=torch.bfloat16),
            w1=torch.empty(1, 1, 1, dtype=torch.uint8),
            w2=torch.empty(1, 1, 1, dtype=torch.uint8),
            topk_weights=torch.empty(1, 1),
            topk_ids=torch.empty(1, 1, dtype=torch.int32),
            activation=None,  # type: ignore[arg-type]
            global_num_experts=1,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=torch.empty(1, 1),
            workspace2=torch.empty(1, 1),
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    assert captured["quant_dtype"] == expected_activation_dtype


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "ocp_mx_scheme",
    [
        OCP_MX_Scheme.w_mxfp4,
        OCP_MX_Scheme.w_mxfp6_e3m2,
        OCP_MX_Scheme.w_mxfp6_e2m3,
    ],
)
def test_ocp_mx_weight_only_input_quantize_is_identity(
    ocp_mx_scheme: OCP_MX_Scheme,
) -> None:
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)

    quantized, scale = moe_kernel_quantize_input(
        A=hidden_states,
        A_scale=None,
        quant_dtype="mxfp4",
        per_act_token_quant=False,
        ocp_mx_scheme=ocp_mx_scheme,
        quantization_emulation=True,
    )

    assert quantized is hidden_states
    assert scale is None


@pytest.mark.cpu_test
def test_ocp_mx_fp8_activation_emulation_qdqs_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    captured: dict[str, object] = {}

    def fake_scaled_fp8_quant(
        A: torch.Tensor,
        A_scale: torch.Tensor | None,
        use_per_token_if_dynamic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured["input"] = A
        captured["scale"] = A_scale
        captured["use_per_token_if_dynamic"] = use_per_token_if_dynamic
        return A.float() + 1, torch.ones(1, dtype=torch.float32)

    def fake_per_tensor_dequantize(
        qA: torch.Tensor,
        qA_scale: torch.Tensor,
    ) -> torch.Tensor:
        captured["dequant_scale"] = qA_scale
        return qA - 1

    monkeypatch.setattr(
        fused_moe_utils.ops,
        "scaled_fp8_quant",
        fake_scaled_fp8_quant,
    )
    monkeypatch.setattr(
        fused_moe_utils,
        "per_tensor_dequantize",
        fake_per_tensor_dequantize,
    )

    quantized, scale = moe_kernel_quantize_input(
        A=hidden_states,
        A_scale=None,
        quant_dtype="mxfp8",
        per_act_token_quant=False,
        quantization_emulation=True,
    )

    assert captured["input"] is hidden_states
    assert captured["scale"] is None
    assert captured["use_per_token_if_dynamic"] is False
    torch.testing.assert_close(quantized, hidden_states)
    assert scale is None


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("mxfp4_backend", "expected_attr"),
    [
        (Mxfp4MoeBackend.EMULATION, "_expert_map"),
        (Mxfp4MoeBackend.TRITON, "_expert_map"),
        (Mxfp4MoeBackend.AITER_MXFP4_BF16, "expert_mask"),
        (Mxfp4MoeBackend.AITER_MXFP4_FP8, "expert_mask"),
    ],
)
def test_ocp_mx_emulation_uses_global_to_local_expert_map(
    mxfp4_backend: Mxfp4MoeBackend,
    expected_attr: str,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.rocm_aiter_fmoe_enabled = True
    layer._expert_map = torch.tensor([-1, 0, 1, -1], dtype=torch.int32)
    layer.expert_mask = torch.tensor([0, 1, 1, 0, 0], dtype=torch.int32)
    layer.quant_method = SimpleNamespace(mxfp4_backend=mxfp4_backend)

    assert layer.expert_map is getattr(layer, expected_attr)


@pytest.mark.cpu_test
def test_ocp_mx_emulation_expert_map_unwraps_modular_method() -> None:
    layer = object.__new__(FusedMoE)
    layer.rocm_aiter_fmoe_enabled = True
    layer._expert_map = torch.tensor([-1, 0, 1, -1], dtype=torch.int32)
    layer.expert_mask = torch.tensor([0, 1, 1, 0, 0], dtype=torch.int32)
    layer.quant_method = SimpleNamespace(
        old_quant_method=SimpleNamespace(mxfp4_backend=Mxfp4MoeBackend.EMULATION)
    )

    assert layer.expert_map is layer._expert_map


@pytest.mark.cpu_test
def test_rocm_aiter_expert_map_defaults_to_global_to_local_map() -> None:
    layer = object.__new__(FusedMoE)
    layer.rocm_aiter_fmoe_enabled = True
    layer._expert_map = torch.tensor([-1, 0, 1, -1], dtype=torch.int32)
    layer.expert_mask = torch.tensor([0, 1, 1, 0, 0], dtype=torch.int32)
    layer.quant_method = SimpleNamespace()

    assert layer.expert_map is layer._expert_map


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    (
        "force_emulation",
        "supports_mx",
        "ocp_mx_scheme",
        "mxfp4_backend",
        "use_rocm_aiter_moe",
        "expected",
    ),
    [
        (
            True,
            True,
            OCP_MX_Scheme.w_mxfp4_a_mxfp4,
            Mxfp4MoeBackend.NONE,
            True,
            True,
        ),
        (
            False,
            True,
            OCP_MX_Scheme.w_mxfp4_a_mxfp4,
            Mxfp4MoeBackend.NONE,
            False,
            True,
        ),
        (
            False,
            True,
            OCP_MX_Scheme.w_mxfp4_a_mxfp4,
            Mxfp4MoeBackend.NONE,
            True,
            False,
        ),
        (
            False,
            False,
            OCP_MX_Scheme.w_mxfp4_a_mxfp4,
            Mxfp4MoeBackend.NONE,
            True,
            True,
        ),
        (
            False,
            True,
            OCP_MX_Scheme.w_mxfp4_a_mxfp6_e3m2,
            Mxfp4MoeBackend.NONE,
            True,
            True,
        ),
        (
            False,
            False,
            OCP_MX_Scheme.w_mxfp4,
            Mxfp4MoeBackend.TRITON,
            False,
            False,
        ),
        (
            False,
            True,
            OCP_MX_Scheme.w_mxfp4,
            Mxfp4MoeBackend.AITER_MXFP4_BF16,
            False,
            True,
        ),
        (
            False,
            False,
            OCP_MX_Scheme.w_mxfp4,
            Mxfp4MoeBackend.AITER_MXFP4_BF16,
            True,
            True,
        ),
        (
            False,
            True,
            OCP_MX_Scheme.w_mxfp4,
            Mxfp4MoeBackend.AITER_MXFP4_BF16,
            True,
            False,
        ),
    ],
)
def test_quark_ocp_mx_emulation_selection(
    force_emulation: bool,
    supports_mx: bool,
    ocp_mx_scheme: OCP_MX_Scheme,
    mxfp4_backend: Mxfp4MoeBackend,
    use_rocm_aiter_moe: bool,
    expected: bool,
) -> None:
    assert (
        _should_emulate_ocp_mx_moe(
            force_emulation=force_emulation,
            supports_mx=supports_mx,
            ocp_mx_scheme=ocp_mx_scheme,
            mxfp4_backend=mxfp4_backend,
            use_rocm_aiter_moe=use_rocm_aiter_moe,
        )
        is expected
    )
