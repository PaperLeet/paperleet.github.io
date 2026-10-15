"""160 real Adam updates on a history-dependent six-item task; not a recommendation benchmark."""
import torch
from .toy import data, encode, scalar_loss, last_item_accuracy_ceiling


def run(causal_item_mask, sampled_next_item_loss):
    histories, targets, initial = data()
    embedding = initial.clone()
    projection = torch.nn.Parameter(torch.zeros(6, 6, dtype=initial.dtype))
    query_projection = torch.nn.Parameter(torch.zeros(6, 6, dtype=initial.dtype))
    optimizer = torch.optim.Adam([query_projection, projection], lr=.1)
    valid = targets != 0
    steps, losses = [], []

    def scores():
        states, attention = encode(histories, embedding, causal_item_mask, query_projection)
        return (states @ projection) @ embedding[1:].T, attention

    def pairs(logits, offset):
        negative = (targets - 1 + offset) % 6
        positive = logits.gather(-1, (targets-1).clamp_min(0)[..., None]).squeeze(-1)
        return positive, logits.gather(-1, negative[..., None]).squeeze(-1)

    def accuracy(logits):
        return float(((logits.argmax(-1)+1 == targets) & valid).sum()) / int(valid.sum())

    with torch.no_grad():
        before, _ = scores()
        p, n = pairs(before, 1)
        initial_loss = float(sampled_next_item_loss(p, n, targets))
    for step in range(160):
        optimizer.zero_grad()
        logits, _ = scores()
        p, n = pairs(logits, step % 5 + 1)
        loss = sampled_next_item_loss(p, n, targets)
        loss.backward(); optimizer.step()
        if step % 10 == 0:
            steps.append(step); losses.append(loss.item())
    with torch.no_grad():
        after, attention = scores()
        p, n = pairs(after, 1)
        final_loss = float(sampled_next_item_loss(p, n, targets))
        expected = scalar_loss(p.tolist(), n.tolist(), targets.tolist())
        original, _ = encode(histories, embedding, causal_item_mask, query_projection)
        changed = histories.clone(); changed[:, -1] = 6
        modified, _ = encode(changed, embedding, causal_item_mask, query_projection)
        future_gap = float((original[:, :-1]-modified[:, :-1]).abs().max())
    steps.append(160); losses.append(final_loss)
    return {'provenance':'PaperLeet SASRec teaching adaptation: two history-dependent sequences plus an empty row, fixed item vectors and identity K/V, learned Q projection/readout and residual LayerNorm. 160 Adam updates on 72 parameters. No FFN/dropout, public dataset or paper benchmark; negatives cycle through non-target items and may have appeared in history.',
            'charts':[{'kind':'line','title':'Sampled next-item loss on training positions','x':steps,'x_label':'Adam update','y_label':'valid-position binary loss','series':[{'name':'learner objective (negative sample rotates)','values':losses}]},
                      {'kind':'bar','title':'First history: causal attention by query','x':['pad','item 1','item 3','item 4'],'x_label':'visible key position','series':[{'name':f'query position {i}','values':row} for i,row in enumerate(attention[0].tolist())]}],
            'metrics':{'updates':160,'last_item_only_upper_bound':last_item_accuracy_ceiling(histories, targets),'trainable_parameters':query_projection.numel()+projection.numel(),'initial_loss':initial_loss,'final_loss':final_loss,
                       'initial_training_accuracy':accuracy(before),'final_training_accuracy':accuracy(after),'maximum_scalar_error':abs(expected-final_loss),
                       'future_state_change':future_gap,'empty_row_attention_mass':float(attention[-1].abs().sum())}}
