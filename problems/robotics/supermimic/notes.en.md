# Implementation notes

A transition's reward precedes its termination gate. Maintain a weight equal to `gamma**t * c_t`, accumulate the reward with that weight, then multiply the weight by `gamma*(1-d_(t+1))`. The remaining weight multiplies the terminal critic value. Horizon rewards sum; only the batch averages.

The actor needs the derivative of soft survival as well as rewards and terminal value. A frozen model still transmits derivatives with respect to its inputs. `detach()` on predicted states, termination or the bootstrap destroys part of the analytic policy gradient.

Critic regression has the opposite boundary: compute targets from detached rewards, termination and critic values, then fit current values to them. Use `V_(t+1)` in the lambda mixture; `V_t` would move the bootstrap backward by one step. With lambda one, the recurrence telescopes to the survival-weighted return, which provides an independent consistency check.

The scalar policy experiment tests backpropagation through a fixed dynamical system. The official method additionally learns a world model from simulator transitions and applies a humanoid tracking reward. Those components are outside this return exercise.
