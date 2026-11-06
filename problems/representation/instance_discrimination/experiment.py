"""Train normalized instance vectors while refreshing their indexed memory rows."""
import math
import torch


def run(instance_nce_loss, update_memory):
    initial = .2 * torch.eye(6, dtype=torch.float64) + .35 * torch.ones((6,6),dtype=torch.float64)
    vectors = torch.nn.Parameter(initial)
    memory = torch.eye(6, dtype=torch.float64)
    ids = torch.arange(6, dtype=torch.int64)
    candidates = torch.tensor([[i] + [j for j in range(6) if j != i] for i in range(6)])
    optimizer = torch.optim.SGD([vectors], lr=.35)
    steps, losses, alignments = [], [], []
    for step in range(81):
        queries = vectors / vectors.norm(dim=1,keepdim=True).clamp_min(1e-12)
        loss = instance_nce_loss(queries, memory, candidates, .2, math.log(math.exp(5)+5))
        if step % 10 == 0:
            steps.append(step); losses.append(float(loss.detach()))
            alignments.append(float((queries.detach()*memory).sum(1).mean()))
        if step < 80:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            memory = update_memory(memory, ids, queries, .95)
    return {'provenance': 'Original six-instance, six-dimensional teaching table; 80 SGD updates (lr .35), T=.2, fixed Z=exp(5)+5, indexed bank EMA .95. All five other toy instances supply noise candidates. It trains no image encoder and estimates no ImageNet result.',
            'charts': [{'kind': 'line', 'title': 'Binary NCE with indexed memory', 'x': steps,
                        'x_label': 'SGD update', 'y_label': 'Training metric',
                        'series': [{'name': 'Binary NCE', 'values': losses},
                                   {'name': 'Query / own-bank cosine', 'values': alignments}]}],
            'metrics': {'updates': 80, 'trainable_parameters': 36,
                        'initial_loss': losses[0], 'final_loss': losses[-1],
                        'initial_alignment': alignments[0], 'final_alignment': alignments[-1],
                        'bank_norm_error': float((memory.norm(dim=1)-1).abs().max()),
                        'bank_requires_grad': memory.requires_grad}}
