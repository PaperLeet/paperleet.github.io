# Understanding a residual block

Where both ReLUs are active, the identity-shortcut Jacobian is `I + W1 @ W2` in the row-vector convention. With a projection it becomes `Ws + W1 @ W2`. With inactive units, insert the corresponding diagonal activation masks; an identity shortcut does not guarantee every gradient is nonzero after the final ReLU.

The scalar oracle computes both layers independently. The derivative checks verify the input, each residual weight and the projection, so a correct forward value cannot hide a detached path.

The six mutants omit the shortcut, omit the final activation, move an extra activation before addition, detach the shortcut, detach the residual, or ignore an explicit projection. Pre-activation blocks can be valid architectures, but they do not implement this exercise's original post-activation contract.

The training graph measures your four-block model's actual loss. It does not compare a fully trained plain network to a convolutional ResNet. Zero initialization of the final branch layer makes the synthetic positive-input starting point easy to inspect; it is a teaching choice rather than a claimed reproduction of the paper's initialization recipe.
