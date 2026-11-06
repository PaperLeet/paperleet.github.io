# SwAV implementation notes

1. The transport matrix is K×B internally. Normalize prototype rows to 1/K before normalizing sample columns to 1/B. Multiply by B before transposing back.
2. Finite iteration count is part of the contract. Sample probability rows sum to one after the last column step; prototype balance is only approximate until convergence.
3. Log-domain normalization avoids overflow. Subtracting a global constant from scores leaves assignments unchanged; arbitrary per-prototype offsets do not generally disappear after finite iterations.
4. Assignment epsilon and prediction temperature are separate. Both matter even if their default magnitudes are similar.
5. Stop the assignment gradients in both APIs. Cross the views in the loss and average the two directions.
6. With soft targets, cross-entropy generally cannot reach zero. Compare it with target entropy to measure the excess KL term.

Native reference calibration: 80 Adam updates, 36 learned parameters, cross-entropy 1.218270 → 1.035732, target-entropy lower bound 1.035710, final excess 2.15e-5. Assignment/scalar errors are below 7e-16. Environment: macOS arm64, Python 3.14.2, PyTorch 2.14.1, CPU float64. Nine mutants cover prototype balancing, batch scaling, assignment epsilon, assignment gradients, view pairing, loss gradients, directional reduction and prediction temperature.

The gradient suite also checks interleaved strided parent tensors, including zero gradients in unused coordinates.
