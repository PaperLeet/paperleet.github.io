# Hints

1. Replace invalid IDs before indexing, then multiply gathered features by the mask. Count valid slots rather than padded width; preserve repeated samples.
2. Clamp the denominator to at least one to obtain zero for an empty neighborhood. Keep even empty operations attached to the gradient graph.
3. Concatenate the previous-hop node feature and mean in that order. Project the concatenation using the supplied row-vector weight layout.
4. Apply ReLU and feature-axis L2 normalization after every hop. Compute all next-hop nodes from the same previous-hop state.
5. Preserve gradients through neighbor features and intermediate hops. Independent scalar finite differences check features and every layer's weights.
6. The supplied training loop fits shared transformations on one graph and freezes them on another. Node IDs only address feature rows; they are not learned parameters.
