# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: per-channel fp8 weight schemes must use kFp8StaticChannelSym
(GroupShape.PER_CHANNEL) so downstream kernel selectors that gate on
``weight_quant_key.scale.group_shape.is_per_channel()`` (e.g. AITER's
preshuffled per-token fp8 GEMM on ROCm) accept the layer instead of
silently falling through to the PyTorch fallback.

Prior to the fix the quark / modelopt / fbgemm schemes used
``kFp8StaticTokenSym`` (PER_TOKEN), and AITER rejected with
``requires per token activation scales and per channel weight scales``.
"""

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8StaticChannelSym,
    kFp8StaticTokenSym,
)


def test_static_channel_sym_has_per_channel_group_shape():
    # Sanity: the constants do what their names say.
    assert kFp8StaticChannelSym.scale.group_shape == GroupShape.PER_CHANNEL
    assert kFp8StaticChannelSym.scale.group_shape.is_per_channel()
    assert not kFp8StaticChannelSym.scale.group_shape.is_per_token()

    assert kFp8StaticTokenSym.scale.group_shape == GroupShape.PER_TOKEN
    assert kFp8StaticTokenSym.scale.group_shape.is_per_token()
    assert not kFp8StaticTokenSym.scale.group_shape.is_per_channel()
