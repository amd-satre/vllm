#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# GSM8K eval for thinking-mode instruct models (Qwen3.6 etc.) without the
# lm-eval "Question:" stop-word trap.
#
# Default gsm8k task config sets until=['Question:','</s>','<|im_end|>'], which
# truncates Qwen3.6-27B's thinking template at 73 chars (it emits
# "- **Question:**" as a bullet label inside <think>). Override the stop list
# to drop "Question:" so the model can finish its CoT.
#
# Run as:
#   HIP_VISIBLE_DEVICES=<id> ./scripts/eval_gsm8k_thinking_safe.sh <model> <run>

set -euo pipefail

MODEL=${1:?model path}
RUN=${2:?run name}
OUT=/workspaces/vllm/eval_results/$RUN
mkdir -p "$OUT"

/workspaces/vllm/.venv/bin/lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,dtype=auto,tensor_parallel_size=1,gpu_memory_utilization=0.85,enforce_eager=False,max_model_len=8192" \
    --include_path /workspaces/vllm/scripts/lm_eval_tasks \
    --tasks gsm8k_thinking \
    --batch_size auto \
    --output_path "$OUT" \
    --log_samples 2>&1 | tee "$OUT/lm_eval.log"
