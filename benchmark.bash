export PYTHONPATH=/workspaces/vllm/tools/local_sitecustomize:/opt/rocm/share/amd_smi:/opt/rocm/lib/python3/site-packages:/opt/rocm/lib/python3.10/site-packages
export HIP_VISIBLE_DEVICES=7
export VLLM_TARGET_DEVICE=rocm

PROFILER_MODE="${PROFILER_MODE:-none}"
BASE_PROFILE_DIR="${BASE_PROFILE_DIR:-/workspaces/vllm/artifacts}"
export VLLM_NVFP4_GEMM_BACKEND=bf16-dequant-gemm

case "$PROFILER_MODE" in
  torch)
    PROFILE_DIR="${PROFILE_DIR:-$BASE_PROFILE_DIR/torch_profile}"
    ;;
  rocprofv3)
    PROFILE_DIR="${PROFILE_DIR:-$BASE_PROFILE_DIR/rocprofv3}"
    ;;
  *)
    PROFILE_DIR="${PROFILE_DIR:-$BASE_PROFILE_DIR/profile}"
    ;;
esac

PROFILE_ARGS=()
if [ "$PROFILER_MODE" = "torch" ]; then
  mkdir -p "$PROFILE_DIR"
  PROFILE_ARGS+=(
    --profile
    --profiler-config.profiler torch
    --profiler-config.torch_profiler_dir "$PROFILE_DIR"
  )
fi

CMD=(
  .venv/bin/python -m vllm.entrypoints.cli.main bench throughput
  --model /amd_models/nvidia/Qwen3-30B-A3B-NVFP4
  --quantization quark
  --trust-remote-code
  --tensor-parallel-size 1
  --dataset-name random
  --input-len 512
  --output-len 128
  --num-prompts 100
  --enforce-eager
  "${PROFILE_ARGS[@]}"
)

if [ "$PROFILER_MODE" = "rocprofv3" ]; then
  mkdir -p "$PROFILE_DIR"
  exec /opt/rocm/bin/rocprofv3 \
    -d "$PROFILE_DIR" \
    -f csv json \
    --runtime-trace \
    --kernel-trace \
    --memory-copy-trace \
    --stats \
    --summary \
    -- "${CMD[@]}"
else
  exec "${CMD[@]}"
fi
