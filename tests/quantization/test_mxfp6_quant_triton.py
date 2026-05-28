# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness + perf check for the fused Triton MXFP6 E2M3 quant+pack
kernel in ``vllm.model_executor.layers.quantization.mxfp6_a4``.

The kernel replaces the existing torch eager-mode chain (used by the
FlyDSL a6w4 dispatch in QuarkOCP_MX) with a single launch. The test
asserts byte-exact equality against the reference chain across the
shapes that the Llama-8B and Qwen-27B linears actually see, then runs
a CUDA-event microbenchmark to make sure the kernel is not slower.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.quantization import mxfp6_a4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kernel requires CUDA/ROCm"
)

# Shapes: (M, K). Cover decode (M=1, padded to 32), small prefill, and a
# couple of mid-range to make sure broadcast indexing scales.
SHAPES = [
    (32, 128),
    (32, 4096),
    (32, 14336),
    (64, 4096),
    (128, 11008),
    (1024, 4096),
]


def _reference(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The existing torch-chain reference inside mxfp6_a4."""
    x_f32 = x_bf16.float().contiguous()
    unpacked, scales = mxfp6_a4._per_token_mxfp6_e2m3(x_f32)
    packed = mxfp6_a4._pack_fp6_e2m3(unpacked)
    return packed, scales


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"M{s[0]}_K{s[1]}")
@pytest.mark.parametrize("block_k", [128])
def test_byte_exact_vs_reference(shape, block_k):
    if not mxfp6_a4._HAS_TRITON:
        pytest.skip("triton missing")
    # Reference chain pulls FlyDSL helpers; skip cleanly if unavailable.
    try:
        mxfp6_a4._load_flydsl_fp4_utils()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"FlyDSL fp4_utils not importable: {e}")

    M, K = shape
    if K % block_k != 0:
        pytest.skip(f"K={K} not divisible by block_k={block_k}")

    torch.manual_seed(0)
    # Use a realistic activation range — RMSNormed output ~N(0, 1) scaled.
    x = (torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.5).to(torch.bfloat16)

    packed_ref, scales_ref = _reference(x)
    packed_kern, scales_kern = mxfp6_a4.per_token_mxfp6_e2m3_packed_triton(
        x, block_k=block_k
    )

    torch.testing.assert_close(scales_kern, scales_ref, atol=0, rtol=0)
    torch.testing.assert_close(packed_kern, packed_ref, atol=0, rtol=0)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"M{s[0]}_K{s[1]}")
def test_speed_not_slower(shape):
    """CUDA-event microbench: kernel must not be slower than the torch chain.

    On AMD/ROCm Triton with HIP events; ``torch.cuda.Event`` works both ways.
    Allows a small slack (1.5×) to cover JIT compile / cache warmup variance.
    """
    if not mxfp6_a4._HAS_TRITON:
        pytest.skip("triton missing")
    try:
        mxfp6_a4._load_flydsl_fp4_utils()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"FlyDSL fp4_utils not importable: {e}")

    M, K = shape
    if K % 128 != 0:
        pytest.skip(f"K={K} not divisible by 128")

    x = (torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.5).to(torch.bfloat16)

    # Warmup both paths
    for _ in range(3):
        _reference(x)
        mxfp6_a4.per_token_mxfp6_e2m3_packed_triton(x)
    torch.accelerator.synchronize()

    def _bench(fn, iters: int = 50) -> float:
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        torch.accelerator.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.accelerator.synchronize()
        return start.elapsed_time(end) / iters  # ms

    t_ref = _bench(lambda: _reference(x))
    t_kern = _bench(lambda: mxfp6_a4.per_token_mxfp6_e2m3_packed_triton(x))

    speedup = t_ref / t_kern
    print(
        f"\n  M={M:5d} K={K:5d}: ref={t_ref * 1000:8.2f} us  "
        f"kern={t_kern * 1000:8.2f} us  speedup={speedup:5.2f}x"
    )

    # Must not be more than 1.5× slower. Expected: many × faster.
    assert t_kern <= t_ref * 1.5, (
        f"kernel is {t_ref / t_kern:.2f}× slower than torch chain at M={M} K={K}"
    )
