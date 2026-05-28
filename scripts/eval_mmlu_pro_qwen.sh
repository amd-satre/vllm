#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MMLU-Pro 5-shot CoT for Qwen3.6-27B (thinking-mode chat template).
# Drops the stock `Question:` stop (which trips Qwen's thinking-mode bullets)
# and bumps max_gen_toks to 8192 to let thinking reasoning complete.
#
# Run as:
#   HIP_VISIBLE_DEVICES=<id> ./scripts/eval_mmlu_pro_qwen.sh <model> <run_name>

set -euo pipefail

MODEL=${1:?model path}
RUN=${2:?run name}
GPU_MEM=${GPU_MEM_UTIL:-0.85}
MAX_LEN=${MAX_MODEL_LEN:-16384}
OUT=/workspaces/vllm/eval_results/$RUN
mkdir -p "$OUT"

/workspaces/vllm/.venv/bin/lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,dtype=auto,tensor_parallel_size=1,gpu_memory_utilization=$GPU_MEM,enforce_eager=False,max_model_len=$MAX_LEN,max_num_seqs=${MAX_NUM_SEQS:-128}" \
    --include_path /workspaces/vllm/scripts/lm_eval_tasks/mmlu_pro \
    --tasks mmlu_pro \
    --apply_chat_template \
    --batch_size auto \
    --output_path "$OUT" \
    --log_samples 2>&1 | tee "$OUT/lm_eval.log"
