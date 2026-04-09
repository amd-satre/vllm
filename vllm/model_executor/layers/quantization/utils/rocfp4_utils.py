# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for RocFP4 quantization.

RocFP4 is temporarily disabled in this environment, so the helper is a no-op.
"""

import torch

__all__ = ["quant_dequant_rocfp4"]

def quant_dequant_rocfp4(
    x: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    return x
