"""Full-batch teaching adaptation, not a reproduction of LearnBPR's bootstrap SGD."""
import torch
from .toy import data, scalar_loss, all_triples, item_only_accuracy_ceiling


def run(eligible_negatives, bpr_mf_loss):
    observed, initial_users, initial_items = data()
    eligible = eligible_negatives(observed, torch.arange(3))
    triples = torch.tensor([[u,i,j] for u,row in enumerate(observed.tolist()) for i,seen in enumerate(row) if seen
                            for j in range(6) if bool(eligible[u,j])],dtype=torch.long).reshape(-1,3)
    users, items = torch.nn.Parameter(initial_users.clone()), torch.nn.Parameter(initial_items.clone())
    optimizer = torch.optim.Adam([users,items],lr=.08)
    l2 = .001
    def measure():
        logits = users @ items.T
        margin = logits[triples[:,0],triples[:,1]] - logits[triples[:,0],triples[:,2]]
        ranked = logits.argsort(-1,descending=True)[:,:2]
        return logits, float((margin>0).sum())/max(1,len(triples)), float(observed.gather(1,ranked).sum())/6, float(margin.min())
    with torch.no_grad():
        before, initial_accuracy, initial_recall, _ = measure()
        initial_loss = float(bpr_mf_loss(users,items,triples,l2))
    steps, losses = [0], [initial_loss]
    for step in range(120):
        optimizer.zero_grad()
        loss = bpr_mf_loss(users,items,triples,l2)
        loss.backward(); optimizer.step()
        if (step+1)%10 == 0:
            with torch.no_grad():
                steps.append(step+1);losses.append(float(bpr_mf_loss(users,items,triples,l2)))
    with torch.no_grad():
        after, final_accuracy, final_recall, minimum_margin = measure()
        final_loss = float(bpr_mf_loss(users,items,triples,l2))
        expected = scalar_loss(users.tolist(),items.tolist(),triples.tolist(),l2)
    return {'provenance':'PaperLeet BPR-MF teaching adaptation: 3 synthetic users, 6 items, 24 observed/unobserved triples, 27 learned factor parameters and 120 full-batch Adam updates. The complete factor-table Gaussian penalty uses l2=.001. This does not reproduce LearnBPR bootstrap SGD, separate per-role regularizers or held-out recommendation benchmarks.',
            'charts':[{'kind':'line','title':'BPR-MF objective during actual optimization','x':steps,'x_label':'Adam update','y_label':'mean pair loss + global prior','series':[{'name':'learner objective','values':losses}]},
                      {'kind':'bar','title':'Learned scores for three conflicting preference groups','x':[f'item {i}' for i in range(6)],'x_label':'catalog item','series':[{'name':f'user {u} likes {2*u}, {2*u+1}','values':row} for u,row in enumerate(after.tolist())]}],
            'metrics':{'updates':120,'trainable_parameters':users.numel()+items.numel(),'triples':len(triples),'eligibility_errors':int((eligible!=(~observed)).sum()),
                       'item_only_accuracy_upper_bound':item_only_accuracy_ceiling(all_triples(observed.tolist())),
                       'initial_loss':initial_loss,'final_loss':final_loss,'initial_pair_accuracy':initial_accuracy,'final_pair_accuracy':final_accuracy,
                       'initial_training_recall_at_2':initial_recall,'final_training_recall_at_2':final_recall,
                       'minimum_final_margin':minimum_margin,'maximum_scalar_error':abs(expected-final_loss)}}
