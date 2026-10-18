# Pitfalls and checks

- True keeps an element; p drops it, so divide by 1-p.
- Do not combine the original test-time scaling convention with inverted dropout.
- Masks are element-wise and must not be silently broadcast or shared across rows.
- Training p=1 needs connected zero, not a detached newly allocated zero.
- The fixed-mask Jacobian is diagonal M/(1-p); preserve its input graph.
- Monte Carlo checks establish sampling behavior, not a generalization advantage.
