# Implementation notes

Each input is already a local token mean: weight it by its token count and divide once by the global count. Empty batches do not dilute the denominator. The experiment compares one update from unequal microbatches with a full-batch update.

Common mistakes:

- Average microbatches equally.
- Omit the global-token denominator.
- Divide again by accumulation steps.
- Return only the final microbatch gradient.

## Why equal batch averaging fails

Two microbatches have mean gradients 1 and 5 and valid-token counts 1 and 3. The correct result is `(1×1+3×5)/4=4`; an equal mean gives 3.

This equals differentiating the sum of all token losses, divided by the total token count. Local mean losses require restoring their token weights during accumulation; do not divide again by accumulation steps. The optimizer should update only once after accumulation.

Inputs here are microbatch gradients for one parameter. A multi-parameter model applies the same weights independently per parameter. For P parameter elements and M microbatches, time is O(MP), with O(P) extra accumulation space.
