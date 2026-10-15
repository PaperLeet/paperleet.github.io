"""Tiny dense full-batch teaching adaptation; only the initial embedding table is learned."""
import torch
from torch.nn import functional as F
from .toy import data, all_triples, scalar_graph, scalar_objective, item_only_accuracy_ceiling


def run(normalized_bipartite_graph, lightgcn_embeddings):
    history, initial = data()
    users, items = history.shape
    graph = normalized_bipartite_graph(history,dtype=initial.dtype)
    triples = torch.tensor(all_triples(history.tolist()),dtype=torch.long)
    table = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([table],lr=.08)
    layers, l2, updates = 2, .001, 160
    def objective():
        final = lightgcn_embeddings(table,graph,layers)
        delta = final[users+triples[:,1]] - final[users+triples[:,2]]
        margin = (final[triples[:,0]] * delta).sum(-1)
        loss = F.softplus(-margin).mean() + l2*table.square().sum()
        return loss,final,margin
    def measure():
        loss,final,margin = objective()
        scores = final[:users] @ final[users:].T
        ranked = scores[:3].argsort(-1,descending=True)[:,:2]
        recall = float(history[:3].gather(1,ranked).sum()) / 6
        return float(loss), scores, float((margin>0).sum())/len(triples), recall, float(margin.min())
    with torch.no_grad():
        initial_loss,_,initial_accuracy,initial_recall,_ = measure()
    steps,losses = [0],[initial_loss]
    for step in range(updates):
        optimizer.zero_grad()
        loss,_,_ = objective()
        loss.backward()
        optimizer.step()
        if (step+1)%10 == 0:
            with torch.no_grad():
                steps.append(step+1);losses.append(float(objective()[0]))
    with torch.no_grad():
        final_loss,scores,accuracy,recall,minimum_margin = measure()
        final = lightgcn_embeddings(table,graph,layers)
        graph_error = max(abs(a-b) for row,wanted in zip(graph.tolist(),scalar_graph(history.tolist())) for a,b in zip(row,wanted))
        isolated_error = max(float((final[n]-table[n]/(layers+1)).abs().max()) for n in (3,users+5))
        expected = scalar_objective(table.tolist(),graph.tolist(),layers,triples.tolist(),users,l2)
    return {'provenance':'PaperLeet LightGCN teaching adaptation: a dense graph with 4 synthetic users, 6 items, 24 observed/unobserved triples, 30 initial embedding parameters and 160 full-batch Adam updates. Uniform aggregation includes layers 0, 1 and 2. The mean BPR term uses a single l2=.001 prior on E0. Scores and Recall@2 describe the training history, not a held-out dataset or paper benchmark. Empty user 3 and item 5 remain isolated.',
            'charts':[{'kind':'line','title':'LightGCN objective during actual optimization','x':steps,'x_label':'Adam update','y_label':'mean pair loss + initial-table prior','series':[{'name':'learner objective','values':losses}]},
                      {'kind':'bar','title':'Scores after two-hop graph propagation','x':[f'item {i}' for i in range(items)],'x_label':'catalog item','series':[{'name':label,'values':scores[u].tolist()} for u,label in enumerate(('user 0 likes 0, 1','user 1 likes 1, 2','user 2 likes 3, 4'))]}],
            'metrics':{'updates':updates,'trainable_parameters':table.numel(),'triples':len(triples),'layers':layers,
                       'initial_loss':initial_loss,'final_loss':final_loss,'initial_pair_accuracy':initial_accuracy,'final_pair_accuracy':accuracy,
                       'initial_training_recall_at_2':initial_recall,'final_training_recall_at_2':recall,'minimum_final_margin':minimum_margin,
                       'item_only_accuracy_upper_bound':item_only_accuracy_ceiling(triples.tolist()),'maximum_graph_error':graph_error,
                       'maximum_isolated_error':isolated_error,'maximum_scalar_error':abs(expected-final_loss)}}
