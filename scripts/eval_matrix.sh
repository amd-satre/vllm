#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Eval matrix: PPL (wikitext) + GSM8K accuracy across (model, dtype) pairs.
#
# Drives lm_eval with vLLM backend. Each run uses one GPU. Results land in
# /workspaces/vllm/eval_results/<run_name>/results.json.
#
# Run as:
#   HIP_VISIBLE_DEVICES=<id> ./scripts/eval_matrix.sh <model_path> <run_name>
# or via the parallel launcher at the bottom of this file.

set -euo pipefail

MODEL=${1:?model path}
RUN=${2:?run name}
OUT=/workspaces/vllm/eval_results/$RUN
mkdir -p "$OUT"

# wikitext PPL + GSM8K 8-shot CoT (standard lm-eval-harness defaults).
# --num_fewshot 8 matches the GSM8K Open LLM Leaderboard config.
# gsm8k_cot uses CoT prompting which matches stronger instruction-tuned baselines.
TASKS="wikitext,gsm8k"

/workspaces/vllm/.venv/bin/lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,dtype=auto,tensor_parallel_size=1,gpu_memory_utilization=0.85,enforce_eager=False,max_model_len=4096" \
    --tasks "$TASKS" \
    --batch_size auto \
    --output_path "$OUT" \
    --log_samples 2>&1 | tee "$OUT/lm_eval.log"
