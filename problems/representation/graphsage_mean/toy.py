"""Original scalar oracles and deterministic unseen-graph fixtures; no downloaded data."""
import math
import random
import torch


def scalar_mean(features, neighbors, valid):
    output=[]
    for ids, mask in zip(neighbors, valid):
        active=[node for node, include in zip(ids, mask) if include]
        output.append([math.fsum(features[node][d] for node in active)/len(active) if active else 0.
                       for d in range(len(features[0]))])
    return output


def scalar_encode(features, neighbors, masks, weights, eps=1e-12):
    state=[row[:] for row in features]
    for index, valid, matrix in zip(neighbors, masks, weights):
        context=scalar_mean(state,index,valid)
        following=[]
        for own, mean in zip(state,context):
            joined=own+mean
            row=[max(0.,math.fsum(joined[j]*matrix[j][d] for j in range(len(joined)))) for d in range(len(matrix[0]))]
            denominator=max(math.sqrt(math.fsum(value*value for value in row)),eps)
            following.append([value/denominator for value in row])
        state=following
    return state


def graph(seed, queries=8, anchors=6):
    """Query features are identically zero; only neighboring anchor types reveal labels."""
    rng=random.Random(seed);kinds=[i%2 for i in range(anchors)]
    features=[[rng.uniform(.8,1.2),rng.uniform(0.,.08),rng.uniform(.05,.15)] if kind==0 else [rng.uniform(0.,.08),rng.uniform(.8,1.2),rng.uniform(.05,.15)] for kind in kinds]+[[0.,0.,0.] for _ in range(queries)]
    labels=[i%2 for i in range(queries)];neighbors=[]
    for node,kind in enumerate(kinds+labels):
        candidates=[i for i,value in enumerate(kinds) if value==kind]
        sampled=[rng.choice(candidates) for _ in range(3)]
        if node>=anchors:
            sampled[-1]=rng.choice([i for i,value in enumerate(kinds) if value!=kind])
            rng.shuffle(sampled)
        neighbors.append(sampled)
    order=list(range(len(features)));rng.shuffle(order);inverse={old:new for new,old in enumerate(order)}
    x=torch.tensor([features[old] for old in order],dtype=torch.float64)
    table=torch.tensor([[inverse[n] for n in neighbors[old]] for old in order],dtype=torch.long)
    valid=torch.ones(table.shape,dtype=torch.bool)
    query=torch.tensor([inverse[anchors+i] for i in range(queries)],dtype=torch.long)
    target=torch.tensor(labels,dtype=torch.long)
    return x,[table,table.clone()],[valid,valid.clone()],query,target


def initial_weights():
    rng=random.Random(403)
    return [torch.tensor([[rng.uniform(.1,.6) for _ in range(out)] for _ in range(inp)],dtype=torch.float64)
            for inp,out in ((6,5),(10,4))]
