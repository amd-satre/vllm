# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for the ``mxfp6_a4`` (FlyDSL MXFP6×MXFP4) quant method.

Covers:
  1. Registration: ``get_quantization_config("mxfp6_a4")`` returns the
     ``Mxfp6A4Config`` class.
  2. ``create_weights`` registers the expected uint8 weight + scale params.
  3. End-to-end ``apply`` on a single Linear with (M=128, N=4096, K=4096)
     matches the bf16 reference to within ``rel_fro < 0.20`` — same bar
     as the standalone validation script in mxfp6_experiments.

These tests require the FlyDSL checkout (``/workspaces/FlyDSL``), the
mxfp6_experiments checkout (``/home/satre/mxfp6_experiments``), and an
MI355X / gfx950 GPU. They are skipped otherwise.

Run:
    .venv/bin/python -m pytest tests/quantization/test_mxfp6_a4.py -v
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_FLYDSL_ROOT = os.environ.get("MXFP6_EXPERIMENTS_FLYDSL_ROOT", "/workspaces/FlyDSL")
_MXFP6_REPO = os.environ.get("MXFP6_EXPERIMENTS_ROOT", "/home/satre/mxfp6_experiments")


def _skip_if_external_deps_missing() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA / ROCm GPU not available")
    if not os.path.isdir(_FLYDSL_ROOT):
        pytest.skip(f"FlyDSL checkout not found at {_FLYDSL_ROOT}")
    if not os.path.isdir(_MXFP6_REPO):
        pytest.skip(f"mxfp6_experiments checkout not found at {_MXFP6_REPO}")
    for p in (_FLYDSL_ROOT, _MXFP6_REPO):
        if p not in sys.path:
            sys.path.insert(0, p)


# ────────────────────────────────────────────────────────────────────────────
# 1. Registration smoke
# ────────────────────────────────────────────────────────────────────────────


def test_registration():
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )
    from vllm.model_executor.layers.quantization.mxfp6_a4 import Mxfp6A4Config

    assert "mxfp6_a4" in QUANTIZATION_METHODS
    assert get_quantization_config("mxfp6_a4") is Mxfp6A4Config


def test_from_config_defaults():
    from vllm.model_executor.layers.quantization.mxfp6_a4 import Mxfp6A4Config

    cfg = Mxfp6A4Config.from_config({})
    assert cfg.group_size == 32
    assert cfg.ignored_layers == []
    assert cfg.get_supported_act_dtypes() == [torch.bfloat16]


# ────────────────────────────────────────────────────────────────────────────
# 2. create_weights / dispatch
# ────────────────────────────────────────────────────────────────────────────


class _FakeLinear(torch.nn.Module):
    """Minimal stand-in for `LinearBase` — `create_weights` only touches
    plain attributes & `register_parameter`."""


def _noop_loader(param, loaded):  # pragma: no cover — never invoked here
    param.data.copy_(loaded)


def test_create_weights_registers_packed_layout(dist_init):
    from vllm.model_executor.layers.quantization.mxfp6_a4 import (
        Mxfp6A4Config,
        Mxfp6A4LinearMethod,
    )

    cfg = Mxfp6A4Config()
    method = Mxfp6A4LinearMethod(cfg, prefix="test.layer")
    layer = _FakeLinear()
    method.create_weights(
        layer,
        input_size_per_partition=4096,
        output_partition_sizes=[4096],
        input_size=4096,
        output_size=4096,
        params_dtype=torch.bfloat16,
        weight_loader=_noop_loader,
    )
    assert layer.weight_packed.shape == (4096, 4096 // 2)
    assert layer.weight_packed.dtype == torch.uint8
    assert layer.weight_scale.shape == (4096, 4096 // 32)
    assert layer.weight_scale.dtype == torch.uint8
    assert layer._mxfp6_a4_can_kernel is True


def test_create_weights_misaligned_K_disables_kernel(dist_init):
    from vllm.model_executor.layers.quantization.mxfp6_a4 import (
        Mxfp6A4Config,
        Mxfp6A4LinearMethod,
    )

    method = Mxfp6A4LinearMethod(Mxfp6A4Config(), prefix="test.misaligned")
    layer = _FakeLinear()
    method.create_weights(
        layer,
        input_size_per_partition=128,  # not divisible by 256
        output_partition_sizes=[256],
        input_size=128,
        output_size=256,
        params_dtype=torch.bfloat16,
        weight_loader=_noop_loader,
    )
    assert layer._mxfp6_a4_can_kernel is False


# ────────────────────────────────────────────────────────────────────────────
# 3. End-to-end apply against bf16 reference
# ────────────────────────────────────────────────────────────────────────────


def _per_block_mxfp4_quant(x_f32: torch.Tensor):
    """Self-contained MXFP4 E2M1 quantizer using FlyDSL fp4_utils via importlib.

    Avoids importing ``scripts.flydsl.bench.utils`` (which would in turn try to
    ``from tests.kernels.utils.fp4_utils import ...`` and clash with vLLM's
    ``tests/kernels`` package namespace under pytest).
    """
    from vllm.model_executor.layers.quantization.mxfp6_a4 import (
        _load_flydsl_fp4_utils,
    )

    fp4u = _load_flydsl_fp4_utils()
    max_normal = 6.0
    blocks = x_f32.unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(dim=-1)
    scale_f32 = (amax / max_normal).clamp_(min=2**-126)
    scales = fp4u.f32_to_e8m0(scale_f32)
    scale_per_elem = fp4u.e8m0_to_f32(scales).repeat_interleave(32, dim=-1)
    x_scaled = (x_f32 / scale_per_elem).contiguous()
    x_q_unpacked = fp4u._f32_to_floatx_unpacked(x_scaled, 2, 1)  # E2M1
    x_q_packed = fp4u.pack_uint4(x_q_unpacked)
    return x_q_packed, scales


def _build_layer_from_bf16_weight(W_bf16: torch.Tensor, prefix: str = "test.oproj"):
    """Run create_weights + populate the uint8 params with the MXFP4 quant of
    the given bf16 weight, then call process_weights_after_loading."""
    _skip_if_external_deps_missing()

    from vllm.model_executor.layers.quantization.mxfp6_a4 import (
        Mxfp6A4Config,
        Mxfp6A4LinearMethod,
    )

    N, K = W_bf16.shape
    method = Mxfp6A4LinearMethod(Mxfp6A4Config(), prefix=prefix)
    layer = _FakeLinear()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[N],
        input_size=K,
        output_size=N,
        params_dtype=torch.bfloat16,
        weight_loader=_noop_loader,
    )

    # Move freshly-created (CPU) params to the device the input lives on, so
    # the dequant fallback and kernel path both see CUDA tensors.
    device = W_bf16.device
    layer.weight_packed = torch.nn.Parameter(
        layer.weight_packed.data.to(device), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        layer.weight_scale.data.to(device), requires_grad=False
    )

    b_packed, b_scales = _per_block_mxfp4_quant(W_bf16.float())
    assert b_packed.shape == layer.weight_packed.shape, (
        f"got {b_packed.shape} expected {layer.weight_packed.shape}"
    )
    layer.weight_packed.data.copy_(b_packed)
    # f32_to_e8m0 returns float8_e8m0fnu — cast via view so the raw byte
    # pattern (biased exponent) lands in the uint8 storage, not a numeric cast.
    layer.weight_scale.data.copy_(b_scales.view(torch.uint8))

    method.process_weights_after_loading(layer)
    return method, layer


@pytest.mark.parametrize("M,N,K", [(128, 4096, 4096)])
def test_apply_matches_bf16_within_tolerance(dist_init, M, N, K):
    _skip_if_external_deps_missing()

    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(0)
    W_bf16 = (
        torch.randn(N, K, device=device, dtype=torch.float32, generator=g) * 0.02
    ).to(torch.bfloat16)
    x_bf16 = (
        torch.randn(M, K, device=device, dtype=torch.float32, generator=g) * 0.5
    ).to(torch.bfloat16)

    method, layer = _build_layer_from_bf16_weight(W_bf16)
    assert layer._mxfp6_a4_can_kernel, (
        "kernel path disabled — check FlyDSL availability"
    )

    y = method.apply(layer, x_bf16, bias=None)
    assert y.shape == (M, N)
    assert y.dtype == torch.bfloat16

    ref = (x_bf16 @ W_bf16.T).float()
    err = (y.float() - ref).norm().item() / max(ref.norm().item(), 1e-12)
    assert err < 0.20, f"rel_fro vs bf16 = {err:.4f} (>= 0.20)"


@pytest.mark.parametrize("M", [1, 7, 16])
def test_apply_falls_back_for_small_M(dist_init, M):
    _skip_if_external_deps_missing()

    device = torch.device("cuda")
    N, K = 4096, 4096
    g = torch.Generator(device=device).manual_seed(0)
    W_bf16 = (
        torch.randn(N, K, device=device, dtype=torch.float32, generator=g) * 0.02
    ).to(torch.bfloat16)
    x_bf16 = (
        torch.randn(M, K, device=device, dtype=torch.float32, generator=g) * 0.5
    ).to(torch.bfloat16)

    method, layer = _build_layer_from_bf16_weight(W_bf16, prefix=f"test.small.M{M}")
    y = method.apply(layer, x_bf16, bias=None)
    assert y.shape == (M, N)
    assert y.dtype == torch.bfloat16
    # Fallback path should be close to bf16 reference (within FP4 quant noise).
    ref = (x_bf16 @ W_bf16.T).float()
    err = (y.float() - ref).norm().item() / max(ref.norm().item(), 1e-12)
    assert err < 0.30, f"M={M} fallback rel_fro = {err:.4f}"
