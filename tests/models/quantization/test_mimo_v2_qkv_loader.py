# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.mimo_v2 import (
    _mimo_v2_copy_presharded_qkv_weight,
)


def _expected_qkv_shard(
    loaded_weight: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    q_heads_per_ckpt_rank = num_heads // num_kv_heads
    ckpt_q_rows = q_heads_per_ckpt_rank * head_dim
    ckpt_k_rows = head_dim
    ckpt_v_rows = v_head_dim
    ckpt_chunk_rows = ckpt_q_rows + ckpt_k_rows + ckpt_v_rows

    def ckpt_shard(
        ckpt_rank: int,
        row_start: int,
        row_count: int,
    ) -> torch.Tensor:
        return loaded_weight.narrow(
            0, ckpt_rank * ckpt_chunk_rows + row_start, row_count
        )

    q_heads_per_rank = num_heads // tp_size
    q_head_start = tp_rank * q_heads_per_rank
    q_head_end = q_head_start + q_heads_per_rank

    q_parts: list[torch.Tensor] = []
    next_q_head = q_head_start
    while next_q_head < q_head_end:
        ckpt_rank = next_q_head // q_heads_per_ckpt_rank
        ckpt_head_start = ckpt_rank * q_heads_per_ckpt_rank
        part_head_end = min(q_head_end, ckpt_head_start + q_heads_per_ckpt_rank)
        part_rows = (part_head_end - next_q_head) * head_dim
        part_start = (next_q_head - ckpt_head_start) * head_dim
        q_parts.append(ckpt_shard(ckpt_rank, part_start, part_rows))
        next_q_head = part_head_end

    if tp_size >= num_kv_heads:
        kv_head_start = tp_rank // (tp_size // num_kv_heads)
        kv_head_count = 1
    else:
        kv_head_count = num_kv_heads // tp_size
        kv_head_start = tp_rank * kv_head_count
    kv_head_end = kv_head_start + kv_head_count

    k_parts = [
        ckpt_shard(ckpt_rank, ckpt_q_rows, ckpt_k_rows)
        for ckpt_rank in range(kv_head_start, kv_head_end)
    ]
    v_parts = [
        ckpt_shard(ckpt_rank, ckpt_q_rows + ckpt_k_rows, ckpt_v_rows)
        for ckpt_rank in range(kv_head_start, kv_head_end)
    ]
    return torch.cat([*q_parts, *k_parts, *v_parts], dim=0)


@pytest.mark.cpu_test
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_mimo_v2_unpaired_presharded_qkv_loader(tp_size: int) -> None:
    config = SimpleNamespace(
        head_dim=2,
        v_head_dim=2,
        num_attention_heads=8,
        num_key_value_heads=4,
    )
    loaded_weight = torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3)

    for tp_rank in range(tp_size):
        expected = _expected_qkv_shard(
            loaded_weight,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            v_head_dim=config.v_head_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        param = torch.nn.Parameter(torch.empty_like(expected))

        assert _mimo_v2_copy_presharded_qkv_weight(
            config=config,
            weight_name="model.layers.0.self_attn.qkv_proj.weight",
            weight_param=param,
            loaded_weight=loaded_weight,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        torch.testing.assert_close(param, expected)


@pytest.mark.cpu_test
def test_mimo_v2_unpaired_presharded_qkv_loader_rejects_other_shapes() -> None:
    config = SimpleNamespace(
        head_dim=2,
        v_head_dim=2,
        num_attention_heads=8,
        num_key_value_heads=4,
    )
    param = torch.nn.Parameter(torch.empty(4, 3))

    assert not _mimo_v2_copy_presharded_qkv_weight(
        config=config,
        weight_name="model.layers.0.self_attn.qkv_proj.weight",
        weight_param=param,
        loaded_weight=torch.empty(31, 3),
        tp_rank=0,
        tp_size=4,
    )
