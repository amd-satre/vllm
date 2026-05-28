# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone wrapper around Quark's ``quantize_quark.py`` for the dense
``qwen3_5`` model_type (e.g. Qwen/Qwen3.6-27B).

Quark ships templates for ``qwen3_5_moe``, ``qwen3_vl_moe``, ``qwen3_next``,
and base ``qwen3`` but no ``qwen3_5`` (dense) entry, so the CLI errors out
with ``no template defined for model type 'qwen3_5'``.

This wrapper registers the missing template before delegating to the shipped
CLI. Per the Quark workflow rule, we do not modify Quark source.

The text decoder is loaded via ``AutoModelForCausalLM`` (i.e.
``Qwen3_5ForCausalLM``); the vision encoder and MTP head are not quantized.
For pure text PPL / GSM8K benchmarks this is the intended scope.

Usage matches quantize_quark.py exactly:

    .venv/bin/python scripts/quark_qwen3_5_ptq.py \\
        --model_dir /amd_models/Qwen/Qwen3.6-27B \\
        --output_dir /amd_models/quark/Qwen3.6-27B-w4a6 \\
        --quant_scheme mxfp4_mxfp6_e2m3 \\
        --num_calib_data 128 --seq_len 512 \\
        --model_export hf_format --data_type auto --device cuda \\
        --exclude_layers lm_head
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from quark.torch import LLMTemplate

QUARK_CLI = Path(
    "/home/satre/Quark/examples/torch/language_modeling/llm_ptq/quantize_quark.py"
)

# Mirrors qwen3_5_moe's template minus the MoE-specific patterns. Excludes:
#  - lm_head:                       output projection, always kept full precision
#  - model.visual.*:                vision tower (silently dropped at load anyway,
#                                   but harmless to list for safety)
#  - mtp.*:                         multi-token-prediction head
#  - *.linear_attn.*:               Mamba-style SSM projections — no MX kernel support
# Register under both the outer multimodal model_type ("qwen3_5") AND the inner
# text-decoder model_type ("qwen3_5_text"). When AutoModelForCausalLM loads a
# multimodal checkpoint it returns the text submodel, whose config.model_type
# is the inner "qwen3_5_text" — that's what Quark's template lookup sees.
_excludes = [
    "lm_head",
    "model.visual.*",
    "mtp.*",
    "*.linear_attn.*",
]
for _mt in ("qwen3_5", "qwen3_5_text"):
    LLMTemplate.register_template(
        LLMTemplate(
            model_type=_mt,
            kv_layers_name=["*k_proj", "*v_proj"],
            q_layer_name="*q_proj",
            exclude_layers_name=_excludes,
        )
    )

# Hand control to the shipped CLI with our args unchanged.
sys.argv[0] = str(QUARK_CLI)
runpy.run_path(str(QUARK_CLI), run_name="__main__")
