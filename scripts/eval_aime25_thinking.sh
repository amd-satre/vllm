#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# AIME25 (30 problems) reproducibility check using thinking-mode chat template.
# Card recommends 32K-82K output tokens; we use 16K to keep runtime under ~20min.
# Sampling matches card recommendation for thinking-mode general tasks.
#
# Run as:
#   HIP_VISIBLE_DEVICES=<id> ./scripts/eval_aime25_thinking.sh <model> <run_name>

set -euo pipefail

MODEL=${1:?model path}
RUN=${2:?run name}
OUT=/workspaces/vllm/eval_results/$RUN
mkdir -p "$OUT"

/workspaces/vllm/.venv/bin/lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,dtype=auto,tensor_parallel_size=1,gpu_memory_utilization=0.85,enforce_eager=False,max_model_len=20480" \
    --include_path /workspaces/vllm/scripts/lm_eval_tasks/aime \
    --tasks aime25_thinking \
    --apply_chat_template \
    --batch_size auto \
    --output_path "$OUT" \
    --log_samples 2>&1 | tee "$OUT/lm_eval.log"
