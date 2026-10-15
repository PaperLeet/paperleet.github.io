# NT-Xent hints

1. Stack A0…A(B-1), B0…B(B-1); swap the two length-B index ranges for targets.
2. Normalize rows, form a Gram matrix and divide by temperature. Mask only the diagonal.
3. Cross entropy targets are partner columns. Reduce by mean over all 2B anchors.
4. A single pair must have zero loss; the two-orthogonal-pair example must retain two negatives per anchor.
5. Both inputs act as anchors and candidates. Keep both derivative paths.
