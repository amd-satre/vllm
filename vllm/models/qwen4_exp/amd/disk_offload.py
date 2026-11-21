# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp PLE table served from the checkpoint files on the host.

The ROCm counterpart of the NVIDIA `disk_offload` backend. Linux only: the files
of the loader's mmap views are found in `/proc/self/maps`.
"""

import mmap
import os
from typing import TYPE_CHECKING

import torch

from vllm.config import get_current_vllm_config
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from ..common.ngram_embedding import Qwen4ExpPLEEmbedding, Qwen4ExpPLEEmbeddingMethod

if TYPE_CHECKING:
    from .ple_layer import Qwen4ExpNGramEmbedding


def _maps_entry(ptr: int) -> tuple[str, int]:
    """Return the path and file offset of `ptr` from `/proc/self/maps`."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            low, high = (int(bound, 16) for bound in fields[0].split("-"))
            if low <= ptr < high:
                path = fields[5].rstrip("\n") if len(fields) > 5 else ""
                return path, ptr - low + int(fields[2], 16)
    return "", 0


class CheckpointRows:
    """Table rows read on the host from the checkpoint's safetensors files.

    The default loader hands over zero-copy mmap views. `add` keeps each shard's
    view until `bind` records its file and offset and releases it, so the table
    is never allocated. Rows stay in the checkpoint format; recently read ones
    are served from the page cache.
    """

    def __init__(self, name: str, num_rows: int, dtype: torch.dtype) -> None:
        self.name = name
        self.num_rows = num_rows
        self.dtype = dtype
        self.row_bytes = 0
        self._shards: dict[int, torch.Tensor] = {}
        self._shard_rows = 0
        self._fds: dict[str, int] = {}
        self._sources: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def bound(self) -> bool:
        return self._sources is not None

    def add(self, index: int, view: torch.Tensor, shard_rows: int) -> None:
        """Keep checkpoint shard `index`, which holds `shard_rows` table rows."""
        if self.bound:
            raise RuntimeError(
                f"{self.name} disk offload does not support weight reload"
            )
        if view.dtype != self.dtype:
            raise ValueError(
                f"{self.name} shard dtype {view.dtype} must match the embedding "
                f"dtype {self.dtype}; disk offload serves rows as stored"
            )
        if index in self._shards:
            raise ValueError(f"Duplicate {self.name} shard {index}")
        self._shards[index] = view.view(torch.uint8)
        self._shard_rows = shard_rows
        self.row_bytes = self._shards[index].shape[1]

    def bind(self) -> None:
        """Record each shard's file and offset and release the loader's views."""
        if self.bound:
            return
        rows = sum(shard.shape[0] for shard in self._shards.values())
        if rows != self.num_rows:
            raise ValueError(f"{self.name} shards cover {rows} of {self.num_rows} rows")
        fds, bases = torch.zeros(2, len(self._shards)).long()
        for index, shard in self._shards.items():
            # The loader hands over mmap views, so find their files here.
            path, offset = _maps_entry(shard.data_ptr())
            if not path or not os.path.isfile(path):
                raise ValueError(
                    f"{self.name} shard {index} is not a file-backed view "
                    f"({path or 'anonymous'})"
                )
            if path not in self._fds:
                self._fds[path] = os.open(path, os.O_RDONLY)
            fds[index], bases[index] = self._fds[path], offset
        self._sources = (fds, bases)
        self._shards.clear()

    def read(self, ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Read the rows of sorted distinct `ids` into the uint8 buffer `rows`."""
        assert self._sources is not None, f"{self.name} rows are not bound"
        assert rows.shape == (ids.numel(), self.row_bytes)
        if ids.numel() and not 0 <= ids[0] <= ids[-1] < self.num_rows:
            raise IndexError(f"{self.name} id out of range for {self.num_rows} rows")
        fds, bases = self._sources
        page, row_bytes = mmap.PAGESIZE, self.row_bytes
        shard = ids // self._shard_rows
        start = bases[shard] + (ids - shard * self._shard_rows) * row_bytes
        first, last = start // page, (start + row_bytes - 1) // page
        run = torch.ones_like(first, dtype=torch.bool)
        run[1:] = (first[1:] > last[:-1] + 1) | (shard[1:] != shard[:-1])
        # Queue every page run before the serial reads.
        for fd, lo, hi in zip(
            fds[shard[run]].tolist(),
            first[run].tolist(),
            last[run.roll(-1)].tolist(),
        ):
            os.posix_fadvise(
                fd, lo * page, (hi - lo + 1) * page, os.POSIX_FADV_WILLNEED
            )
        for row, fd, offset in zip(rows.numpy(), fds[shard].tolist(), start.tolist()):
            if os.preadv(fd, [row], offset) != row_bytes:
                raise ValueError(
                    f"{self.name} shard file {os.readlink(f'/proc/self/fd/{fd}')} "
                    f"is truncated at offset {offset}"
                )


class Qwen4ExpPLEDiskOffloadEmbedding(Qwen4ExpPLEEmbedding):
    """PLE rows read on the host from the checkpoint shard files.

    No table is allocated. Each step, `stage_checkpoint_rows` reads the rows
    of that step's n-gram ids from the safetensors files and copies them to a
    fixed device buffer, which the forward reads, so graphs replay it.
    Every ETP rank stages full rows, so no ETP reduction is needed.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            data_parallel_rank=data_parallel_rank,
            reduce_results=reduce_results,
        )
        self._dummy = get_current_vllm_config().load_config.load_format == "dummy"
        device, pin = self.weight.device, self.weight.is_cuda
        tokens = (max_total_tokens, num_ngram_heads)
        self._host_ids = torch.empty(
            tokens, dtype=torch.long, device="cpu", pin_memory=pin
        )
        # Device staging, pinned host rows (written only after the step's stream
        # sync, so after the previous H2D) and the distinct rows' bytes.
        shape, dtype = (*tokens, embedding_dim), self.weight.dtype
        self._staging = torch.zeros(shape, dtype=dtype, device=device)
        self._host = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin)
        self._distinct = torch.empty_like(self._host).view(torch.uint8).flatten(0, 1)
        self._checkpoint_rows = CheckpointRows("PLE weight", self.org_vocab_size, dtype)

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate no rows; they stay in the checkpoint files."""
        return torch.empty(0, embedding_dim, dtype=dtype)

    def accept_checkpoint_shard(
        self, shard_index: int, loaded_weight: torch.Tensor, shard_size: int
    ) -> None:
        """Keep a checkpoint shard as a zero-copy byte view of its storage."""
        self._checkpoint_rows.add(shard_index, loaded_weight, shard_size)

    def bind_file_shards(self) -> None:
        """Record each shard's file and offset and release the loader's views."""
        if not self._dummy:
            self._checkpoint_rows.bind()

    def copy_ids_to_host(self, ids: torch.Tensor) -> None:
        self._host_ids[: ids.shape[0]].copy_(ids, non_blocking=True)

    def stage_rows(self, num_tokens: int) -> None:
        """Read the rows of `_host_ids[:num_tokens]` and copy them to staging."""
        host = self._host[:num_tokens]
        if self._dummy:
            host.view(torch.uint8).zero_()
        else:
            ids = self._host_ids[:num_tokens].flatten()
            ids, inverse = ids.unique(return_inverse=True)
            distinct = self._distinct[: ids.numel()]
            self._checkpoint_rows.read(ids, distinct)
            out = host.flatten(0, 1).view(torch.uint8)
            torch.index_select(distinct, 0, inverse, out=out)
        self._staging[:num_tokens].copy_(host, non_blocking=True)

    def staged_rows(self, num_tokens: int) -> torch.Tensor:
        """Return this step's staged rows, flattened to one row per token."""
        return self._staging[:num_tokens].flatten(-2)

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        return None

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        raise RuntimeError("PLE disk offload does not support prefetch")


def stage_checkpoint_rows(
    modules: list[tuple["Qwen4ExpNGramEmbedding", Qwen4ExpPLEDiskOffloadEmbedding]],
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
) -> None:
    """Stage disk-offload PLE rows for one step behind one stream sync.

    The ids hash this step's tokens, which the previous step samples on the
    GPU, so the host read has to wait for them.
    """
    for module, embedding in modules:
        ids = module.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        embedding.copy_ids_to_host(ids)
    with gpu_sync_allowed():
        torch.accelerator.current_stream().synchronize()
    for _, embedding in modules:
        embedding.stage_rows(input_ids.shape[0])
