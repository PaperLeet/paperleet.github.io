# Implementation notes

The output describes the role of the **target**, not the predictor. A user token can precede a supervised assistant token; an assistant token can precede an unsupervised user token. Check both directions in a multi-turn example. Do not shift the returned mask and then shift it again in the loss.

Only adjacent conversation-ID equality matters. Do not compare every token with column zero: that would lose later packed conversations. ID magnitude, batch order and the number of turns are irrelevant. At a packed boundary, the new conversation's first token lacks a predictor from that conversation, even if its role is assistant. At left truncation, the first retained token is similarly excluded. Right truncation does not erase earlier targets or create a missing EOS.

Validity is a separate input. A target and its immediate predecessor must both be real tokens; metadata under padding is deliberately untrusted. This matters when EOS and PAD use the same token ID. Supplied assistant EOS has role 2, so it remains eligible; the mask does not stop at the first assistant response or first EOS. Headers/BOS are assigned role 0 upstream.

The boolean computation is O(BT), with O(BT) output storage. Empty B/T dimensions need no special token. The reference writes only a newly allocated boolean result; it never modifies caller tensors. Integer masks have no derivative contract. The gradient test verifies the effect of using the mask in a supplied loss, not a gradient through role IDs.

## Source scope

The TRL SFT documentation establishes assistant-only supervision and next-token label alignment. Its packing guide describes multiple strategies and truncation. Our numeric roles, explicit conversation IDs, EOS ownership and predecessor rule form an original, bounded teaching interface. We do not claim that this function reproduces TRL preprocessing or makes attention isolation unnecessary.

## Experiment and acceptance

Two fixed rows contain 24 token positions and eleven valid targets. Each predictor owns four trainable float64 logits, initially zero. SGD uses learning rate 4 for exactly 80 updates; no sampling or data download occurs. The standard seed is 42 but this probe makes no random draws. Reference and learner calls execute the same loop.

An independent hand-labelled mask defines evaluation NLL and excluded parameters. Starting NLL is ln(4); correct runs reach target NLL below .08 and accuracy 1, with exactly zero change to every excluded logit. The final tolerance leaves room above the deterministic reference result while requiring useful optimization. There is no held-out dataset or shared language model here: the experiment demonstrates which parameters receive training, not generalization. All plotted losses are measured from the optimizer trajectory.

The four mutants use predictor roles, omit conversation boundaries, omit padding, or supervise column zero. Each changes one rule and is caught by direct assertions. In particular, learning a low loss on a wrong mask is insufficient: independent target loss, mask disagreements and excluded-parameter changes are checked too.

Calibration on 2026-10-17: Python 3.14.2 / native Torch 2.14.1, macOS arm64, produced final target NLL `0.02795611111943972`, accuracy `1.0`, and excluded-logit change `0.0`. Three native reference/mutation runs passed (120 test cases, all four mutants caught). The unchanged exported suite also passed in Pyodide 0.29.4 / NumPy 2.2.5: 8 reference tests, four assertion-caught mutants and 8 learner tests, including the same 80-step experiment. This records correctness on these runners, not a universal latency guarantee.
