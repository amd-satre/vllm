#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# GSM8K 8-shot CoT (gsm8k_cot, Meta-style Q:/A: prompt) — matches what
# Llama 3.1 model card uses. Llama card reports GSM8K 8-shot CoT = 84.5
# for Llama-3.1-8B-Instruct.
#
# Run as:
#   HIP_VISIBLE_DEVICES=<id> ./scripts/eval_gsm8k_cot8.sh <model> <run>

set -euo pipefail

MODEL=${1:?model path}
RUN=${2:?run name}
OUT=/workspaces/vllm/eval_results/$RUN
mkdir -p "$OUT"

/workspaces/vllm/.venv/bin/lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,dtype=auto,tensor_parallel_size=1,gpu_memory_utilization=0.85,enforce_eager=False,max_model_len=4096" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size auto \
    --output_path "$OUT" \
    --log_samples 2>&1 | tee "$OUT/lm_eval.log"
