"""120 real optimizer steps on four synthetic two-token queries; no BERT training."""
import torch
from .toy import data, pooled_pair_gap, scalar_scores


def run(maxsim_scores, pairwise_loss):
    initial, documents, qm, dm, positive, negative = data()
    query = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([query], lr=.08)
    rows = torch.arange(4)
    curves, steps = [], []
    with torch.no_grad():
        before = maxsim_scores(query, documents, qm, dm)
        initial_loss = pairwise_loss(before[rows, positive], before[rows, negative]).item()
    for step in range(120):
        optimizer.zero_grad()
        scores = maxsim_scores(query, documents, qm, dm)
        loss = pairwise_loss(scores[rows, positive], scores[rows, negative])
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            curves.append(loss.item()); steps.append(step)
    with torch.no_grad():
        after = maxsim_scores(query, documents, qm, dm)
        final_loss = pairwise_loss(after[rows, positive], after[rows, negative]).item()
    steps.append(120); curves.append(final_loss)
    expected = scalar_scores(query.tolist(), documents.tolist(), qm.tolist(), dm.tolist())
    error = max(abs(value - wanted) for row, target in zip(after.tolist(), expected) for value, wanted in zip(row, target))

    def recall(values):
        return sum(max(range(8), key=lambda index: row[index]) == target for row, target in zip(values.tolist(), positive.tolist())) / 4

    margin_before = (before[rows, positive] - before[rows, negative]).tolist()
    margin_after = (after[rows, positive] - after[rows, negative]).tolist()
    return {'provenance': 'ColBERT teaching adaptation: 4 synthetic queries / 8 fixed documents / 4 features / 120 Adam steps on query vectors. No BERT, tokenizer, ANN index, pretrained weights or paper benchmark.',
            'charts': [
                {'kind':'line','title':'Pairwise loss during query-vector optimization','x':steps,'x_label':'Adam update','y_label':'mean pairwise cross-entropy','series':[{'name':'learner objective','values':curves}]},
                {'kind':'line','title':'Relevant vs blended-negative score margin','x':list(range(4)),'x_label':'synthetic query','y_label':'positive minus negative','series':[{'name':'before','values':margin_before},{'name':'after','values':margin_after}]},
            ],
            'metrics': {'initial_loss':initial_loss,'final_loss':final_loss,'initial_recall_at_1':recall(before),'final_recall_at_1':recall(after),
                        'minimum_final_margin':min(margin_after),'maximum_scalar_error':error,'pooled_pair_gap':pooled_pair_gap(documents, dm),'updates':120}}
