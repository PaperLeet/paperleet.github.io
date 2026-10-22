# Implementation notes

Successful coder output with easy=0.8 and c=0.5 earns 2.3; both render and syntax failure earn -0.15. For solver alpha=0.9, agreement with invalid format earns 0.9; disagreement with valid format earns 0.1.

The coder does not inherit the proposer’s easy-score cap. Failed renders have no visual scores; error penalties can add. The binary difficulty range annotation near Eq.11 conflicts with Eq.7; this exercise uses the continuous min(c,1-c) from Eq.7. Solver format is additive, not a gate.

The reference behavior is defined by the starter, including deterministic tie handling and caller-supplied upstream observations. The toy policy is a controlled test, not evidence that model-generated labels are correct. Compare a held-out gold check separately from agreement with synthetic labels.

https://arxiv.org/abs/2603.09206

The [official Coder reward implementation](https://github.com/zli12321/MM-Zero/blob/main/MM-zero_final/reward_function/codegen_reward.py) also uses continuous `min(c, 1-c)`. This exercise isolates the paper reward and the two stated error penalties; repository-specific wrong-render-type penalties are outside its API.
