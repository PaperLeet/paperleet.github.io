# Pitfalls and checks

- Real and fake labels are opposite; softplus signs determine the update direction.
- Add two independently averaged expectations without another factor of one half.
- The minimax generator is a legitimate alternative outside the selected non-saturating contract.
- Avoid sigmoid followed by log at extreme logits.
- Detach generated samples before D during D steps, not D logits inside the loss. Freeze D weights during G steps while keeping input gradients.
