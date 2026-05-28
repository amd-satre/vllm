# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.quantization.quark.utils import (
    check_equal_or_regex_match,
    should_ignore_layer,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("layer_name", "patterns"),
    [
        ("model.layers.0.self_attn.q_proj", ["*self_attn*"]),
        ("model.layers.12.mlp.gate", ["*mlp.gate"]),
        ("model.layers.0.mlp.down_proj", ["model.layers.0.mlp.*"]),
        ("lm_head", ["*lm_head"]),
    ],
)
def test_quark_ignore_matches_export_globs(
    layer_name: str,
    patterns: list[str],
) -> None:
    assert check_equal_or_regex_match(layer_name, patterns)


@pytest.mark.cpu_test
def test_quark_ignore_matches_fused_layer_export_globs() -> None:
    assert should_ignore_layer(
        "model.layers.0.self_attn.qkv_proj",
        ignore=["*self_attn*"],
        fused_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    )


@pytest.mark.cpu_test
def test_quark_ignore_still_rejects_partial_fused_matches() -> None:
    with pytest.raises(ValueError, match="different quantization schemes"):
        should_ignore_layer(
            "model.layers.0.self_attn.qkv_proj",
            ignore=["*q_proj"],
            fused_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
        )


@pytest.mark.cpu_test
def test_quark_ignored_non_linear_layer_returns_no_quant_method() -> None:
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    config = QuarkConfig({"exclude": ["*self_attn*"]})

    assert (
        config.get_quant_method(
            object(),  # type: ignore[arg-type]
            "model.layers.0.self_attn",
        )
        is None
    )


@pytest.mark.cpu_test
def test_quark_detects_exclude_aware_per_block_fp8() -> None:
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    config = QuarkConfig({"exclude": []})
    weight_config = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_block",
        "is_dynamic": False,
        "block_size": [128, 128],
    }
    input_config = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_group",
        "is_dynamic": True,
        "group_size": 128,
    }

    assert config._is_fp8_w8a8(weight_config, input_config)

    input_config["is_dynamic"] = False
    assert not config._is_fp8_w8a8(weight_config, input_config)
