# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for the FlyDSL dispatch inside ``QuarkOCP_MX``.

Covers the (mxfp4 weight, mxfp6_e2m3 activation) combo that today falls into
the Python QDQ emulation path. With ``VLLM_MX_USE_FLYDSL=1`` and a gfx950
GPU + FlyDSL checkout, ``apply_weights`` must dispatch to the FlyDSL
a6w4 kernel and produce a result within ``rel_fro < 0.30`` of the bf16
reference.

Requires the FlyDSL checkout (``/workspaces/FlyDSL``) and the
mxfp6_experiments checkout (``/home/satre/mxfp6_experiments``).
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


def _per_block_mxfp4_quant(x_f32: torch.Tensor):
    """Self-contained MXFP4 E2M1 quantizer using FlyDSL fp4_utils via importlib.

    See tests/quantization/test_mxfp6_a4.py:_per_block_mxfp4_quant for the
    rationale (vLLM ``tests/kernels/`` namespace clash with FlyDSL).
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


def _fake_quant_spec_w_mxfp4_a_mxfp6_e2m3():
    """Quark-style quant spec dicts for (weight=fp4, input=fp6_e2m3)."""
    weight_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
        "is_dynamic": False,
    }
    input_spec = {
        "dtype": "fp6_e2m3",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
        "is_dynamic": True,
    }
    return weight_spec, input_spec


@pytest.fixture(autouse=True)
def _enable_flydsl(monkeypatch):
    monkeypatch.setenv("VLLM_MX_USE_FLYDSL", "1")
    # The module-level flag is read at import time; force-refresh it for
    # this test by patching the module attribute directly.
    from vllm.model_executor.layers.quantization.quark.schemes import (
        quark_ocp_mx as q,
    )

    monkeypatch.setattr(q, "_VLLM_MX_USE_FLYDSL", True)
    yield


def _build_scheme():
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
        QuarkOCP_MX,
    )

    weight_spec, input_spec = _fake_quant_spec_w_mxfp4_a_mxfp6_e2m3()
    scheme = QuarkOCP_MX(weight_spec, input_spec)
    # In real vLLM runs the model sets default dtype to bf16 before the scheme
    # is constructed; we mirror that here so out_dtype matches the kernel.
    scheme.out_dtype = torch.bfloat16
    return scheme


def test_init_routes_w_mxfp4_a_mxfp6_e2m3_to_flydsl(dist_init):
    _skip_if_external_deps_missing()
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import on_gfx950

    if not (current_platform.is_rocm() and on_gfx950()):
        pytest.skip("FlyDSL dispatch only enabled on gfx950")

    scheme = _build_scheme()
    assert scheme._use_flydsl, (
        "FlyDSL dispatch should engage on gfx950 with VLLM_MX_USE_FLYDSL=1"
    )
    assert scheme.emulate is False, "FlyDSL path should bypass emulate"


@pytest.mark.parametrize("M,N,K", [(128, 4096, 4096)])
def test_apply_matches_bf16_within_tolerance(dist_init, M, N, K):
    _skip_if_external_deps_missing()
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import on_gfx950

    if not (current_platform.is_rocm() and on_gfx950()):
        pytest.skip("FlyDSL dispatch only enabled on gfx950")

    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(0)
    W_bf16 = (
        torch.randn(N, K, device=device, dtype=torch.float32, generator=g) * 0.02
    ).to(torch.bfloat16)
    x_bf16 = (
        torch.randn(M, K, device=device, dtype=torch.float32, generator=g) * 0.5
    ).to(torch.bfloat16)

    scheme = _build_scheme()

    # Mimic the layer attributes that create_weights would populate; we skip
    # create_weights itself because the FlyDSL dispatch only reads layer.weight
    # and layer.weight_scale.
    layer = torch.nn.Module()
    layer.logical_widths = [N]
    layer.input_size_per_partition = K
    layer.output_size_per_partition = N

    b_packed, b_scales = _per_block_mxfp4_quant(W_bf16.float())
    layer.weight = torch.nn.Parameter(b_packed.to(device), requires_grad=False)
    # E8M0 returns float8_e8m0fnu; .view(uint8) keeps the raw exponent byte.
    layer.weight_scale = torch.nn.Parameter(
        b_scales.view(torch.uint8).to(device), requires_grad=False
    )

    scheme.process_weights_after_loading(layer)
    y = scheme.apply_weights(layer, x_bf16, bias=None)

    assert y.shape == (M, N)
    assert y.dtype == torch.bfloat16

    ref = (x_bf16 @ W_bf16.T).float()
    err = (y.float() - ref).norm().item() / max(ref.norm().item(), 1e-12)
    assert err < 0.30, f"rel_fro vs bf16 = {err:.4f} (>= 0.30)"


@pytest.mark.parametrize("M", [1, 7, 16])
def test_apply_pads_M_to_32(dist_init, M):
    """Decode (M=1) and small prefill chunks must succeed via M-pad."""
    _skip_if_external_deps_missing()
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import on_gfx950

    if not (current_platform.is_rocm() and on_gfx950()):
        pytest.skip("FlyDSL dispatch only enabled on gfx950")

    device = torch.device("cuda")
    N, K = 4096, 4096
    g = torch.Generator(device=device).manual_seed(0)
    W_bf16 = (
        torch.randn(N, K, device=device, dtype=torch.float32, generator=g) * 0.02
    ).to(torch.bfloat16)
    x_bf16 = (
        torch.randn(M, K, device=device, dtype=torch.float32, generator=g) * 0.5
    ).to(torch.bfloat16)

    scheme = _build_scheme()
    layer = torch.nn.Module()
    layer.logical_widths = [N]
    layer.input_size_per_partition = K
    layer.output_size_per_partition = N

    b_packed, b_scales = _per_block_mxfp4_quant(W_bf16.float())
    layer.weight = torch.nn.Parameter(b_packed.to(device), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        b_scales.view(torch.uint8).to(device), requires_grad=False
    )

    scheme.process_weights_after_loading(layer)
    y = scheme.apply_weights(layer, x_bf16, bias=None)

    assert y.shape == (M, N)
    assert y.dtype == torch.bfloat16
    ref = (x_bf16 @ W_bf16.T).float()
    err = (y.float() - ref).norm().item() / max(ref.norm().item(), 1e-12)
    assert err < 0.35, f"M={M} pad rel_fro = {err:.4f} (>= 0.35)"
