# Multi-turn SFT: Assistant Targets and Conversation Boundaries

## 1. Problem

Build the target mask for several assistant turns, including when conversations share a packed row. [TRL documents assistant-only loss](https://huggingface.co/docs/trl/sft_trainer#train-on-assistant-messages-only) and [next-token label shifting](https://huggingface.co/docs/trl/sft_trainer#label-shifting-and-masking). This exercise supplies structured token metadata instead of asking you to implement a tokenizer or chat template.

Implement:

```python
assistant_target_mask(role_ids: Tensor, conversation_ids: Tensor,
                      attention_mask: Tensor) -> Tensor
```

Return a **bool Tensor `[B,T]` at target positions**, on the input device. It is not an attention matrix, shifted mask, token-ID array or loss. Preserve both dimensions and leave all inputs unchanged. No gradients are needed for integer/boolean metadata.

## 2. Core rule

Let `r`, `c` and `a` be the role, conversation ID and validity arrays. Define

$$
m_{b,0}=\mathrm{False},\qquad
m_{b,j}=[r_{b,j}=2]\land a_{b,j}\land a_{b,j-1}\land[c_{b,j}=c_{b,j-1}],\quad j\ge1.
$$

The predecessor must exist in the same conversation, but **its role need not be assistant**: the last user/context position can predict the first assistant token. Return the mask without shifting it. A supplied causal loss later pairs `logits[:, :-1]`, `tokens[:, 1:]` and `mask[:, 1:]` exactly once.

## 3. From metadata to code

All inputs have the same `[B,T]` shape. Compare target positions with their immediate predecessors, combine the role/validity/boundary conditions, and leave the first column false. B or T may be zero; singleton dimensions remain present.

For one row with all positions valid:

```text
role:          [1, 2, 2, 1, 2, 2, 2, 2]
conversation:  [7, 7, 7, 7, 7, 7, 9, 9]
target mask:   [F, T, T, F, T, T, F, T]
```

Both assistant turns in conversation 7 contribute. Position 6 begins a different packed conversation and cannot be predicted by position 5, even though both are assistant tokens.

## 4. Exercise conventions

- CPU int64 `role_ids` and `conversation_ids`; bool `attention_mask`. Shapes match, with `B,T>=0`. Inputs are valid; validation is not part of the required function. Strided inputs work.
- At valid positions roles are **0=system/header**, **1=user**, **2=assistant content**, **3=tool**. Chat headers and BOS are marked 0 by preprocessing. Valid conversation IDs are nonnegative int64, need not be consecutive, and may be reused in another batch row. All turns of a conversation share its ID; adjacent packed conversations use different IDs.
- `attention_mask=True` means a real, retained token. Padding can appear at either end or inside a row. Metadata under padding may contain arbitrary int64 values and must not affect the result. A real token immediately after padding has no valid immediate predictor and is excluded.
- **EOS ownership is supplied**, not inferred from token IDs. An EOS ending an assistant response has role 2 and is eligible under the same rule. User/tool EOS retains that turn's role. The same token ID can also serve as padding; only the validity mask distinguishes it here. An assistant EOS does not end the entire conversation or suppress later assistant turns.
- **Truncation happens before this function.** At a left cut, column zero has no retained predictor, even if it is inside an assistant response. Later visible assistant targets can still contribute. At a right cut, retain eligible targets through the last visible token; do not invent an EOS or discard a partial answer merely because its EOS was cut away.
- Roles, EOS ownership, ID layout and the exact boundary formula are explicit PaperLeet teaching conventions. [TRL's packing/truncation guide](https://huggingface.co/docs/trl/reducing_memory_usage#packing) motivates tracking sequence boundaries; this function does not reproduce every TRL packing strategy.

This is a **loss-target mask**, not protection against attention crossing conversations. A packed model needs separate attention/position handling. The exercise does not implement that handling, chat-template parsing, tokenization or language-model training.

## 5. Experiment

<viz id="toy_experiment" />

The supplied experiment has two original 12-token rows, four vocabulary classes and eleven hand-labelled assistant targets. It includes several turns, assistant EOS, a packed boundary, a left-truncated assistant segment and a padding gap. Your returned mask controls 80 SGD updates of independently parameterized position logits.

Charts show actual training NLL, NLL on the independently labelled targets, and one row's target mask. Correct selection learns the eleven targets while excluded predictor logits stay exactly unchanged. The initial four-way uniform NLL is `log(4)`; the accepted final target NLL is below `.08`. These are deterministic synthetic optimization results, not held-out language quality or a pretrained-model benchmark. Reference Lab previews call the solution; submitted artifacts call your function.

## 6. Run and verify

```bash
paperleet test foundations/multiturn_sft_mask --impl problems/foundations/multiturn_sft_mask/solution.py
paperleet mutate foundations/multiturn_sft_mask
paperleet ci --problem foundations/multiturn_sft_mask
```

T0 checks shape/empty inputs; T1 uses hand-labelled turns; T2 checks an exhaustive local truth table, truncation, padding metadata, strides, ID/row independence and analytic next-token gradients. T3 trains with the submitted mask. Native Torch and CPU/WASM use the same tests, 80 updates and tolerances.

## 7. Further work

Feed this mask into the existing `foundations/sft_loss_mask` loss as its response mask, together with the original validity mask. That function shifts targets once. Then explain separately what a tokenizer must label, what a causal loss must shift, and what a packed attention implementation must isolate. Those are distinct responsibilities.
