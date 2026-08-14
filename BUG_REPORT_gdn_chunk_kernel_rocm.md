# Qwen3.6 (hybrid Gated-Delta-Net) bf16 generation collapses into a fixed degenerate
phrase on ROCm — traced to `chunk_gated_delta_rule`'s prefill-time final-state output

## Summary

On this AMD ROCm build of vLLM, running `Qwen/Qwen3.6-27B` or `Qwen/Qwen3.6-35B-A3B`
(both hybrid Gated-Delta-Net + attention architectures) in **bf16** produces
byte-for-byte identical degenerate output on a large fraction of prompts that require
longer chain-of-thought: generation emits a fixed stub —

```
\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User Input:**\n   - **
```

— verbatim, across dozens of unrelated GSM8K math problems, then stops (hits an
`until`/EOS-like condition almost immediately after). Short completions (where the
model doesn't need extended reasoning) are unaffected and produce correct, coherent
output.

This single bug was responsible for GSM8K accuracy on the *bf16* baseline looking far
lower than several **quantized** derivatives of the same checkpoints (which route
through different linear kernels and don't hit it) — an apparent "quantization
improves reasoning" result that is actually an artifact of this bug corrupting the
unquantized baseline.

## Environment

- vLLM: `0.23.1.dev1+g9ddef7117.d20260715`
- Hardware: AMD MI350X (gfx950/CDNA4), ROCm build
- Models: `Qwen/Qwen3.6-27B` (dense) and `Qwen/Qwen3.6-35B-A3B` (MoE) — both hybrid
  Gated-Delta-Net + full-attention architectures (`Qwen3_5ForConditionalGeneration`)
- `dtype=bfloat16`, both `enforce_eager=True` and default (CUDA-graph + `torch.compile`)
  modes affected identically

## Reproduction

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3.6-27B", dtype="bfloat16", trust_remote_code=True,
           max_model_len=4096)
prompt = ("Question: A robe takes 2 bolts of blue fiber and half that much white "
          "fiber. How many bolts in total does it take?\nAnswer:")
out = llm.generate([prompt], SamplingParams(max_tokens=300, temperature=0.0))
print(out[0].outputs[0].text)
```

Observed (broken):
```
\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User Input:**\n   - **
```
(and nothing further — generation halts)

At scale: running `lm-evaluation-harness`'s `gsm8k` task (5-shot, `max_gen_toks=2048`,
greedy) against this model, ~35-40% of a 20-example sample collapse into this exact
stub, and 2/20 produced *zero-length* completions. This reproduces identically on both
the dense 27B model and the 35B-A3B MoE model — **not architecture-family-specific
beyond "has a Gated-Delta-Net mixer."**

## Root cause (localized, not fully fixed)

Disabling exactly one custom op resolves the issue completely:

```python
llm = LLM(..., compilation_config={"custom_ops": ["-chunk_gated_delta_rule"]})
```

With this, the identical prompt above produces a fully coherent, correct trace ending
in the right answer. This was also independently confirmed by running the *reference*
(non-Triton) implementation via `transformers` on CPU with the same weights — fully
coherent, correct output, no collapse.

Tracing `chunk_gated_delta_rule`'s only call site
(`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,
`QwenGDNLinearAttention._forward_core`) shows it is invoked **only for the prefill
portion** of a request:

```python
(
    core_attn_out_non_spec,
    last_recurrent_state,
) = self.chunk_gated_delta_rule(
    q=query_non_spec, k=key_non_spec, v=value_non_spec,
    g=g_non_spec, beta=beta_non_spec,
    initial_state=initial_state, output_final_state=True,
    cu_seqlens=attn_metadata.prefill_query_start_loc,
    chunk_indices=attn_metadata.chunk_indices,
    chunk_offsets=attn_metadata.chunk_offsets,
    use_qk_l2norm_in_kernel=False,
)
# Init cache
ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)
```

`ssm_state` is the persistent recurrent-state cache that every subsequent **decode**
step reads as `initial_state` (via the separate `fused_sigmoid_gating_delta_rule_update`
decode kernel — a different code path, unaffected by the op-disable workaround). This
precisely explains the failure signature we observed:

- Short completions barely depend on the carried recurrent state → look correct.
- Longer decode runs increasingly depend on a state that, if numerically wrong from
  the start, compounds error over many autoregressive steps → eventual collapse into a
  degenerate, high-probability attractor state and early stop.
- The bug is present in **eager mode** as well as compiled/CUDA-graph mode → it is not
  a graph-capture/state-aliasing issue introduced by `torch.compile`, but something in
  `chunk_gated_delta_rule`'s own Triton (FLA) implementation on this platform, most
  likely specifically in the **final recurrent state** it exports (`last_recurrent_state`)
  rather than the immediate output logits for the prefilled tokens (since those look
  fine on short completions).

We have **not** identified the exact incorrect line inside the FLA Triton kernel
(`vllm/model_executor/layers/fla/ops/chunk.py` and friends — the actual
`chunk_gated_delta_rule` implementation is a large, general-purpose flash-linear-
attention kernel, not AMD-specific: `GDN_AITER_TRITON_AVAILABLE` was `False` in this
environment, so this exact code path also executes on NVIDIA CUDA — it is unclear
whether this is ROCm-specific miscompilation of an otherwise-correct kernel, or a
latent bug that happens not to manifest on NVIDIA hardware/Triton versions). One
observation that may help narrow it further: `torch.ops.vllm.qwen_gdn_attention_core`
(the custom op one level up, wrapping the whole conv1d + recurrent-attention block)
declares `mutates_args=["a_or_z_out", "core_attn_out"]`
(`direct_register_custom_op(...)` near the bottom of `qwen_gdn_linear_attn.py`) — this
does not appear to include the `ssm_state`/recurrent-state buffer that
`chunk_gated_delta_rule` reads and (via the outer function) writes. We have **not**
verified this is the actual mechanism (it's a plausible lead given custom-op mutation
declarations affect how PyTorch's dispatcher and tracing treat repeated calls, but we
did not trace deeply enough to confirm a specific incorrect codegen/caching decision
resulting from it), so it should be treated as a hypothesis for someone more familiar
with this kernel to check, not a confirmed mechanism.

## Suggested next steps for whoever picks this up

1. Add a unit test comparing `chunk_gated_delta_rule`'s Triton output (`o`,
   `final_state`) against the pure-PyTorch reference (`transformers`' fallback
   implementation, or a naive recurrence) for a batch of random inputs, run on
   ROCm — check both `o` (looks probably-correct given short completions work) and
   specifically `final_state` (our suspected culprit) for divergence.
2. Check whether `direct_register_custom_op`'s `mutates_args` for
   `qwen_gdn_attention_core` (and any registration for `chunk_gated_delta_rule`
   itself, if it goes through a similar custom-op path) fully and correctly declares
   every buffer that is read-and-written across the prefill/decode state handoff.
3. If reproducible on NVIDIA CUDA too, this is not ROCm-specific and should be
   retitled/rescoped accordingly — we could not test on NVIDIA hardware in this
   environment.

## Verified workaround

```python
compilation_config={"custom_ops": ["-chunk_gated_delta_rule"]}
```

This forces the plain (slower) fallback path and produces correct, coherent output
in all cases we tested. Confirmed via a 150-example CPU ground-truth cross-check
(reference `transformers` implementation) that the *aggregate* GSM8K score impact of
this bug was small for both models tested, despite individual generations being
visibly corrupted — so benchmark numbers gathered without the workaround are only
mildly (not wildly) distorted in aggregate, but per-example outputs should not be
trusted without it.
