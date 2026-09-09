# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quark W4A6 (MXFP4 weight x MXFP6-E2M3 activation) routing.

Config-level only: asserts that a checkpoint exported by Quark's
``mxfp4_mxfp6_e2m3`` scheme is recognized as OCP-MX and lands on the a6w4
branch of :class:`QuarkOCP_MX`. No GPU and no aiter required -- the FlyDSL
kernel itself is probed separately by ``is_flydsl_a6w4_supported()`` and simply
falls back to the existing high-precision emulation when absent.
"""

import pytest

from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_Scheme

# What Quark's MXFP4_MXFP6E2M3Scheme serializes into quantization_config:
# OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False) for the weight and
# OCP_MXFP6E2M3Spec(ch_axis=-1, is_dynamic=True) for the input, both carrying
# OCP_MXSpec.OCP_MX_SPEC_KWARGS.
_OCP_MX_COMMON = {
    "qscheme": "per_group",
    "group_size": 32,
    "scale_format": "e8m0",
    "scale_type": "float",
    "round_method": "half_even",
    "symmetric": None,
    "ch_axis": -1,
}
QUARK_W4A6_WEIGHT = {"dtype": "fp4", "is_dynamic": False, **_OCP_MX_COMMON}
QUARK_W4A6_INPUT = {"dtype": "fp6_e2m3", "is_dynamic": True, **_OCP_MX_COMMON}


def _config() -> QuarkConfig:
    return QuarkConfig(
        quant_config={
            "global_quant_config": {
                "weight": QUARK_W4A6_WEIGHT,
                "input_tensors": QUARK_W4A6_INPUT,
            },
            "layer_quant_config": {},
            # _find_matched_config consults this before falling back to
            # global_quant_config; a real export always carries it.
            "layer_type_quant_config": {},
            "kv_cache_group": [],
        },
        kv_cache_group=[],
    )


def test_quark_w4a6_is_ocp_mx():
    """A Quark w4a6 export must pass the OCP-MX gate that selects QuarkOCP_MX."""
    assert _config()._is_w_ocp_mx_a_x(QUARK_W4A6_WEIGHT, QUARK_W4A6_INPUT)


def test_quark_w4a6_dtype_normalization():
    """Quark's fp4/fp6_e2m3 must normalize to the mx* names the scheme uses."""
    assert QUARK_W4A6_WEIGHT["dtype"].replace("fp", "mxfp") == "mxfp4"
    assert QUARK_W4A6_INPUT["dtype"].replace("fp", "mxfp") == "mxfp6_e2m3"
    assert (
        OCP_MX_Scheme.from_quant_dtype("mxfp6_e2m3", "mxfp4")
        is OCP_MX_Scheme.w_mxfp4_a_mxfp6_e2m3
    )


def test_quark_w4a6_selects_a6w4_branch():
    """QuarkOCP_MX built from a Quark w4a6 spec sets is_a6w4."""
    QuarkOCP_MX = pytest.importorskip(
        "vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx",
        reason="QuarkOCP_MX import requires aiter",
    ).QuarkOCP_MX

    scheme = QuarkOCP_MX(QUARK_W4A6_WEIGHT, QUARK_W4A6_INPUT)
    assert scheme.weight_dtype == "mxfp4"
    assert scheme.input_dtype == "mxfp6_e2m3"
    assert scheme.is_a6w4
    # use_flydsl_a6w4 is is_a6w4 AND the kernel probe, so it must never be set
    # without the kernel actually being importable -- otherwise a machine with
    # no aiter a6w4 kernel would dispatch to a custom op that was never
    # registered instead of falling back to emulation.
    from vllm.model_executor.layers.quantization.flydsl_a6w4_linear import (
        is_flydsl_a6w4_supported,
    )

    assert scheme.use_flydsl_a6w4 == is_flydsl_a6w4_supported()


def test_w4a6_excluded_attention_stays_bf16():
    """Excluded self_attn layers must not be re-quantized to dynamic MXFP4.

    Regression test: this predicate reads the *raw* Quark config, where the
    activation dtype is serialized as "fp6_e2m3". Comparing it against the
    normalized "mxfp6_e2m3" never matches, which silently sends every excluded
    attention projection through dynamic-MXFP4 emulation.
    """
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )

    cfg = _config()
    cfg.quant_config["exclude"] = ["re:.*self_attn.*"]
    cfg.dynamic_mxfp4_quant = True

    layer = LinearBase.__new__(LinearBase)  # only isinstance() is consulted
    method = cfg.get_quant_method(layer, "model.layers.0.self_attn.qkv_proj")
    assert isinstance(method, UnquantizedLinearMethod)


def test_non_w4a6_excluded_attention_still_overridden():
    """The exemption must be scoped to w4a6, not to OCP-MX generally.

    mxfp6_e3m2 activations keep the pre-existing dynamic-MXFP4 override, so the
    normalization above must not widen into a substring match on "fp6".
    """
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )

    cfg = _config()
    cfg.quant_config["global_quant_config"]["input_tensors"] = {
        **_OCP_MX_COMMON,
        "dtype": "fp6_e3m2",
        "is_dynamic": True,
    }
    cfg.quant_config["exclude"] = ["re:.*self_attn.*"]
    cfg.dynamic_mxfp4_quant = True

    layer = LinearBase.__new__(LinearBase)
    method = cfg.get_quant_method(layer, "model.layers.0.self_attn.qkv_proj")
    assert not isinstance(method, UnquantizedLinearMethod)


@pytest.mark.parametrize(
    "weight_dtype,input_dtype",
    [("fp4", "fp4"), ("fp4", "fp6_e3m2"), ("fp6_e2m3", "fp6_e2m3")],
)
def test_non_w4a6_schemes_untouched(weight_dtype, input_dtype):
    """Neighbouring OCP-MX schemes must not be pulled onto the a6w4 branch."""
    QuarkOCP_MX = pytest.importorskip(
        "vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx",
        reason="QuarkOCP_MX import requires aiter",
    ).QuarkOCP_MX

    scheme = QuarkOCP_MX(
        {**_OCP_MX_COMMON, "dtype": weight_dtype, "is_dynamic": False},
        {**_OCP_MX_COMMON, "dtype": input_dtype, "is_dynamic": True},
    )
    assert not scheme.is_a6w4
    assert not scheme.use_flydsl_a6w4
