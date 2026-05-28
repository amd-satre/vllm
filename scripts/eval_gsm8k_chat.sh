#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
MODEL=${1:?model path}
RUN=${2:?run name}
OUT=/workspaces/vllm/eval_results/$RUN
mkdir -p "$OUT"
/workspaces/vllm/.venv/bin/lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,dtype=auto,tensor_parallel_size=1,gpu_memory_utilization=0.85,enforce_eager=False,max_model_len=8192" \
    --tasks gsm8k \
    --apply_chat_template \
    --gen_kwargs "max_gen_toks=2048" \
    --batch_size auto \
    --output_path "$OUT" \
    --log_samples 2>&1 | tee "$OUT/lm_eval.log"
