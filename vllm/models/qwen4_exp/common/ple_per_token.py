# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-independent PLE embedding for the ``per_token`` and ``zero`` modes.

Shared by the AMD and NVIDIA Qwen4Exp PLE layers; the hashed n-gram table
(``ngram`` mode) stays in the platform-specific modules.
"""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen4ExpTextConfig

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from .ple_mode import PLE_EMBEDDING_MODE_PER_TOKEN, PLE_EMBEDDING_MODE_ZERO

logger = init_logger(__name__)

# Checkpoint tensors of the n-gram embedding that the per-token / zero modes
# do not use; they are skipped without error.
NGRAM_ONLY_BUFFERS = frozenset(
    {"layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"}
)


class PerTokenTableMethod(QuantizeMethodBase):
    """Post-load hook: fail loudly if ``per_token_table.weight`` was not loaded.

    The table is allocated uninitialized, and vLLM's generic "all weights were
    loaded" check is skipped for quantized checkpoints, so without this a
    checkpoint lacking the tensor would load silently and serve garbage.
    """

    requires_device_loading: bool = False
    supports_pre_processed_weights: bool = True

    def create_weights(self, layer: nn.Module, *weight_args, **extra_weight_attrs):
        raise NotImplementedError("the per-token table is created by its owner")

    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("the per-token table is applied by its owner")

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer.check_table_loaded()


class Qwen4ExpPerTokenEmbedding(nn.Module):
    """PLE embedding without the n-gram hash table (``per_token``/``zero``).

    ``per_token``: ``forward`` returns ``per_token_table[input_ids]``, where the
    table is a plain, TP-replicated parameter of shape
    ``[vocab_size, embedding_dim]`` (checkpoint tensor name
    ``<prefix>.per_token_table.weight``). Ids are clamped into
    ``[0, vocab_size - 1]`` on device (no host sync, safe under CUDA graphs and
    torch.compile) instead of raising: an out-of-range id reads the first or
    last row.

    ``zero``: no table exists; ``forward`` returns zeros of shape
    ``[num_tokens, embedding_dim]`` in ``params_dtype`` on ``input_ids.device``.

    ``query_start_loc`` and ``ngram_context`` are accepted for interface
    compatibility with :class:`Qwen4ExpNGramEmbedding` and ignored.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        mode: str,
        *,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if mode not in (PLE_EMBEDDING_MODE_PER_TOKEN, PLE_EMBEDDING_MODE_ZERO):
            raise ValueError(f"Unsupported PLE embedding mode {mode!r}")
        self.mode = mode
        self.embedding_dim = int(embedding_dim)
        self.vocab_size = int(config.vocab_size)
        self.params_dtype = (
            params_dtype if params_dtype is not None else torch.get_default_dtype()
        )
        self.table_loaded = False
        if mode == PLE_EMBEDDING_MODE_PER_TOKEN:
            self.quant_method = PerTokenTableMethod()
            # nn.Embedding gives the checkpoint name per_token_table.weight;
            # the weight is overwritten by load_weights, so skip random init.
            self.per_token_table = nn.Embedding(
                self.vocab_size,
                self.embedding_dim,
                _weight=torch.empty(
                    self.vocab_size, self.embedding_dim, dtype=self.params_dtype
                ),
                _freeze=True,
            )
        logger.info(
            "Initialized PLE embedding in %s mode (vocab=%d, dim=%d, dtype=%s)",
            mode,
            self.vocab_size,
            self.embedding_dim,
            self.params_dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del query_start_loc, ngram_context
        input_ids = input_ids.reshape(-1)
        if self.mode == PLE_EMBEDDING_MODE_ZERO:
            return torch.zeros(
                (input_ids.shape[0], self.embedding_dim),
                dtype=self.params_dtype,
                device=input_ids.device,
            )
        ids = input_ids.long().clamp(0, self.vocab_size - 1)
        return F.embedding(ids, self.per_token_table.weight)

    def dequantize(
        self, embeddings: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        del output_dtype
        return embeddings

    def check_table_loaded(self) -> None:
        """Raise unless the per-token table was loaded from the checkpoint."""
        if self.mode != PLE_EMBEDDING_MODE_PER_TOKEN:
            return
        weight = self.per_token_table.weight
        if not self.table_loaded:
            raise ValueError(
                "ple_embedding_mode='per_token' but the checkpoint has no "
                "tensor '...ple_embedding.per_token_table.weight' "
                f"(expected shape {(self.vocab_size, self.embedding_dim)}, "
                f"dtype {self.params_dtype}). Export a per-token checkpoint or "
                "use ple_embedding_mode='ngram'/'zero'."
            )
        if tuple(weight.shape) != (self.vocab_size, self.embedding_dim):
            raise ValueError(
                f"per_token_table.weight has shape {tuple(weight.shape)}, expected "
                f"{(self.vocab_size, self.embedding_dim)}"
            )
        logger.info(
            "PLE per_token_table loaded=True shape=%s dtype=%s",
            tuple(weight.shape),
            weight.dtype,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``per_token_table.weight``; skip the n-gram-only tensors."""
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if (
                leaf_name.startswith("hashstats_")
                or leaf_name == "token_lookup"
                or name in NGRAM_ONLY_BUFFERS
                or name.startswith("ngram_embedding.")
            ):
                continue
            if name == "per_token_table.weight":
                if self.mode == PLE_EMBEDDING_MODE_ZERO:
                    continue
                weight = self.per_token_table.weight
                if weight.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(weight.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                if loaded_weight.dtype not in (torch.bfloat16, weight.dtype):
                    raise ValueError(
                        f"dtype mismatch for {name}: expected bfloat16 or "
                        f"{weight.dtype}, got {loaded_weight.dtype}"
                    )
                with torch.no_grad():
                    weight.copy_(
                        loaded_weight.to(device=weight.device, dtype=weight.dtype)
                    )
                self.table_loaded = True
                loaded.add(name)
                continue
            raise ValueError(
                f"Unexpected PLE embedding tensor {name!r} in "
                f"ple_embedding_mode={self.mode!r}"
            )
        return loaded


__all__ = ["NGRAM_ONLY_BUFFERS", "PerTokenTableMethod", "Qwen4ExpPerTokenEmbedding"]
