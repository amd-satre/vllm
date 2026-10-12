# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Engram replacement modes (zero / global_mean / per_token)."""
import importlib.util
import os
import tempfile

import numpy as np
import pytest
import torch

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "vllm", "models", "deepseek_v41", "common", "engram_modes.py"
)
_spec = importlib.util.spec_from_file_location("engram_modes_under_test", _PATH)
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)

NC, DIM, CV = 24, 256, 50  # n_hash_cols, head_dim, (toy) compressed vocab


def _tables(tmp, layer=1):
    g = torch.randn(NC, DIM); u = torch.randn(CV, NC, DIM)
    np.save(os.path.join(tmp, f"global_mean_L{layer}.npy"), g.numpy())
    np.save(os.path.join(tmp, f"U_L{layer}_bf16.npy"), u.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
    return g, u.to(torch.bfloat16)


@pytest.mark.parametrize("tp", [1, 2, 3, 5, 8])
def test_shard_geometry_and_local_columns(tp):
    parts = [em.ReplacementShardInfo(NC, DIM, tp, r) for r in range(tp)]
    assert all(p.part_n_hash_cols == -(-NC // tp) for p in parts)
    assert [p.head_start for p in parts] == [r * parts[0].part_n_hash_cols for r in range(tp)]
    with tempfile.TemporaryDirectory() as tmp:
        _, u = _tables(tmp)
        rebuilt = []
        for p in parts:
            rep = em.EngramReplacement("per_token", 1, p, CV, tmp)
            assert rep.U.shape == (CV, p.part_n_hash_cols, DIM)
            rebuilt.append(rep.U)
        full = torch.cat(rebuilt, dim=1)[:, :NC]  # all-gather then crop, as Engram.embed does
        assert torch.equal(full, u)


def test_modes_fill_rows():
    with tempfile.TemporaryDirectory() as tmp:
        g, u = _tables(tmp)
        shard = em.ReplacementShardInfo(NC, DIM, 1, 0)
        T = 7; ids = torch.randint(0, CV, (T,))
        rows = torch.full((T, NC, DIM), 3.0, dtype=torch.bfloat16)
        em.EngramReplacement("zero", 1, shard, CV, None).fill(rows, None)
        assert not rows.any()
        rows = torch.empty(T, NC, DIM, dtype=torch.bfloat16)
        em.EngramReplacement("global_mean", 1, shard, CV, tmp).fill(rows, None)
        assert torch.equal(rows, g.to(torch.bfloat16).unsqueeze(0).expand(T, -1, -1))
        em.EngramReplacement("per_token", 1, shard, CV, tmp).fill(rows, ids)
        assert torch.equal(rows, u[ids])


def test_zero_lookup_leaves_residual_untouched():
    """Reference Engram forward (inference/model.py, bias-free wkv): E = 0 -> kv = 0 -> value = 0, so the branch adds exactly 0
    even though the gate is ~0.5 (signed-sqrt of the clamped dot)."""
    torch.manual_seed(0)
    dim, hc, ncol = 64, 4, NC
    wkv = torch.randn(dim * (hc + 1), ncol * DIM) * 0.02
    q_w, k_w = torch.randn(hc, dim), torch.randn(hc, dim)
    h = torch.randn(5, hc, dim); eps, clamp = 1e-20, 1e-6
    E = torch.zeros(5, ncol, DIM)
    kv = E.flatten(-2) @ wkv.T
    key, value = kv.split([hc * dim, dim], dim=-1)
    key = key.unflatten(-1, (hc, dim))
    weight = q_w * k_w
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * weight * key).sum(-1) * rstd * dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp).sqrt(), dot))
    out = h + gate.unsqueeze(-1) * value.unsqueeze(-2)
    assert torch.isfinite(gate).all() and torch.allclose(gate, torch.full_like(gate, 0.5 + 0.0), atol=1e-3)
    assert torch.equal(out, h)
