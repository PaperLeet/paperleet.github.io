# Two variance estimators and two execution modes

Training normalizes with the variance of the observed batch, denominator `m`. Estimating population variance from independently sampled batches requires the `m/(m-1)` correction. Using that correction in the output changes Algorithm 1; omitting it from moving variance changes the selected inference estimator.

For one channel let `z=(x-mean)/sqrt(v+eps)` and `q=upstream*gamma`. The input derivative is:

$$\frac{\partial L}{\partial x_i}=\frac{q_i-\operatorname{mean}(q)-z_i\operatorname{mean}(qz)}{\sqrt{v+\epsilon}}.$$

Detaching moments removes the two correction terms. The scale and bias derivatives sum `upstream*z` and `upstream` over batch/spatial positions. Running estimates, by contrast, are intentionally detached so later batches do not backpropagate through earlier ones.

The six mutants confuse training and estimation denominators, omit the running correction, reverse the EMA coefficient, detach live moments, move epsilon outside the square root or retain a graph in running state. The fixed inference companion test checks a structural property, not a model-accuracy claim. Moving variance from finite random batches need not equal 4 exactly; the experiment plots the estimate without grading an unsupported exact-population promise.
