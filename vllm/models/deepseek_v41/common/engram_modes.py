# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental replacements of the Engram TABLE LOOKUP (zero / global_mean / per_token).

Only the lookup output is replaced: the rows handed to ``wkv`` have the same shape and
dtype as the real lookup ([num_tokens, part_n_hash_cols, head_dim] bf16, this rank's head
columns); wkv, the gate, the norms and the residual add run unchanged. Pure torch, no vLLM
imports, so the logic is unit-testable on CPU.
"""

import os

import numpy as np
import torch
from torch import nn

MODES = ("ngram", "zero", "global_mean", "per_token")


class ReplacementShardInfo(nn.Module):
    """Stand-in for ParallelEngramEmbedding in replacement modes: carries the shard
    geometry the Engram module reads (no parameters, no host/GPU table)."""

    def __init__(self, n_hash_cols: int, head_dim: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.n_hash_cols = n_hash_cols
        self.dim = head_dim
        self.tp_size = tp_size
        self.dp_size = 1
        self.cpu_offload = False
        self.part_n_hash_cols = -(-n_hash_cols // tp_size)
        self.head_start = tp_rank * self.part_n_hash_cols


def _local_columns(t: torch.Tensor, col_dim: int, head_start: int, part: int) -> torch.Tensor:
    """This rank's [head_start, head_start + part) hash columns along ``col_dim``,
    zero-padded when the last shard overruns the real column count (the padding is
    cropped after the TP all-gather, exactly like the real lookup's)."""
    n = t.shape[col_dim]
    sl = t.narrow(col_dim, head_start, max(0, min(part, n - head_start)))
    if sl.shape[col_dim] < part:
        pad_shape = list(sl.shape)
        pad_shape[col_dim] = part - sl.shape[col_dim]
        sl = torch.cat([sl, sl.new_zeros(pad_shape)], dim=col_dim)
    return sl.contiguous()


class EngramReplacement(nn.Module):
    """Stand-in for one Engram layer's lookup. ``fill(rows, comp_ids)`` writes
    ``rows`` = staged_rows[:T] in place."""

    def __init__(self, mode: str, layer_id: int, shard: ReplacementShardInfo, comp_vocab: int,
                 table_dir: str | None, device: torch.device | str = "cpu"):
        super().__init__()
        assert mode in MODES and mode != "ngram", mode
        self.mode, self.layer_id = mode, layer_id
        self.comp_vocab = comp_vocab
        part, hs, nc, dim = shard.part_n_hash_cols, shard.head_start, shard.n_hash_cols, shard.dim
        self.gm = None
        self.U = None
        if mode == "global_mean":
            g = torch.from_numpy(np.load(os.path.join(table_dir, f"global_mean_L{layer_id}.npy")))
            assert tuple(g.shape) == (nc, dim), (tuple(g.shape), (nc, dim))
            self.gm = _local_columns(g.to(torch.bfloat16), 0, hs, part).to(device)
        elif mode == "per_token":
            u = torch.from_numpy(np.load(os.path.join(table_dir, f"U_L{layer_id}_bf16.npy")).view(np.uint16).astype(np.int16)).view(torch.bfloat16)
            assert tuple(u.shape) == (comp_vocab, nc, dim), (tuple(u.shape), (comp_vocab, nc, dim))
            self.U = _local_columns(u, 1, hs, part).to(device)

    def fill(self, rows: torch.Tensor, comp_ids: torch.Tensor | None) -> None:
        if self.mode == "zero":
            rows.zero_()
        elif self.mode == "global_mean":
            rows.copy_(self.gm.unsqueeze(0).expand_as(rows))
        else:
            assert comp_ids is not None and comp_ids.shape[0] == rows.shape[0]
            torch.index_select(self.U, 0, comp_ids, out=rows)
