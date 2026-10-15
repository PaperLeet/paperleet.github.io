# Common pitfalls

Normalize each token, not a document-wide flattened tensor. Max reduces document tokens; sum reduces query tokens. Mean pooling, averaging query evidence and raw unnormalized dot products are different scoring rules.

Zero-valued padding can beat legitimate negative similarities. Exclude document padding before max. An all-masked document has no candidate; handle it without creating NaN gradients, then apply the invalid-document score. Absent query positions contribute zero. Original query-augmentation `[MASK]` embeddings are valid encoded positions and are not automatically removed.

Use the first valid exact maximum for the specified tie gradient. Do not detach document vectors: offline representation storage does not mean the encoders should have no training gradients. Positive rescaling invariance assumes the norms stay above epsilon.

The stable paired objective depends on negative minus positive. A shared shift cancels, increasing positive evidence lowers loss, and extreme margins must not overflow. Tests check scalar finite differences independently of the tensor scorer and differentiate both score inputs.

The 120-step experiment learns only tiny query vectors against frozen synthetic documents. Its labeled Recall@1 and collapsed pooling example are instructional; they are not BERT or MS MARCO results.
