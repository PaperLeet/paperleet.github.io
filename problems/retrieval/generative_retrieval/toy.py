"""Fixed conditional toy model; not trained TIGER weights or a relevance benchmark."""
import itertools
import math
import torch


def catalog():
    # Same eight items as semantic_ids/toy.py; cross-package consistency is tested.
    return torch.tensor([[1,2,0,0],[1,2,0,1],[1,2,0,2],[4,2,0,0],
                         [2,4,0,0],[3,2,1,0],[3,0,2,0],[0,0,0,0]], dtype=torch.long)


def probabilities(prefix):
    fixed = {(): [.05,.36,.10,.25,.24],
             (1,): [.04,.04,.50,.04,.38],
             (1,2): [.45,.05,.10,.10,.30],
             (1,2,0): [.30,.25,.20,.15,.10],
             (3,): [.20,.05,.60,.05,.10]}
    peaks = {(3,2):(1,.8), (3,2,1):(0,.8), (4,):(2,.8), (4,2):(0,.8),
             (4,2,0):(0,.8), (2,):(4,.7), (2,4):(0,.8), (2,4,0):(0,.8),
             (3,0):(2,.7), (3,0,2):(0,.8), (0,):(0,.8), (0,0):(0,.8), (0,0,0):(0,.8)}
    if prefix in fixed:
        return fixed[prefix]
    if prefix in peaks:
        token, p = peaks[prefix]
        result = [(1-p)/4] * 5
        result[token] = p
        return result
    return [.025,.025,.025,.025,.9]


def next_logits(prefix):
    return torch.tensor([math.log(p) for p in probabilities(prefix)], dtype=torch.float64)


def exhaustive():
    scores = []
    for tokens in itertools.product(range(5), repeat=4):
        # Scalar normalized logs; independent of torch.log_softmax and beam pruning.
        score = 0.
        for depth, token in enumerate(tokens):
            ps = probabilities(tokens[:depth])
            score += math.log(ps[token]) - math.log(sum(ps))
        scores.append((tokens, score))
    return sorted(scores, key=lambda item: (-item[1],item[0]))
