#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Offline throughput benchmark via `vllm bench throughput`.
# Random dataset: 500 prompts of (input=1024, output=512).
#
# Run as:
#   HIP_VISIBLE_DEVICES=<id> ./scripts/bench_throughput.sh <model> <run>

set -euo pipefail

MODEL=${1:?model path}
RUN=${2:?run name}
OUT=/workspaces/vllm/eval_results/throughput/$RUN
mkdir -p "$OUT"

/workspaces/vllm/.venv/bin/vllm bench throughput \
    --model "$MODEL" \
    --dtype auto \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --dataset-name random \
    --input-len 1024 \
    --output-len 512 \
    --num-prompts 500 \
    --output-json "$OUT/result.json" 2>&1 | tee "$OUT/bench.log"
