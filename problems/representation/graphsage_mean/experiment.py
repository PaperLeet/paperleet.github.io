"""Actual small supervised optimization and inductive evaluation on an unseen synthetic graph."""
import torch
from torch.nn import functional as F
from .toy import graph, initial_weights


def run(graphsage_encode):
    train=graph(17,queries=8,anchors=6);held_out=graph(83,queries=10,anchors=8)
    weights=[torch.nn.Parameter(value) for value in initial_weights()]
    classifier=torch.nn.Parameter(torch.zeros(4,2,dtype=torch.float64))
    bias=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
    optimizer=torch.optim.Adam([*weights,classifier,bias],lr=.04)
    def measure(data):
        x,neighbors,masks,query,target=data
        embeddings=graphsage_encode(x,neighbors,masks,weights)
        logits=embeddings[query]@classifier+bias
        return F.cross_entropy(logits,target),float((logits.argmax(-1)==target).sum())/len(target)
    with torch.no_grad():initial_loss,initial_accuracy=measure(train);initial_loss=float(initial_loss)
    steps,losses,accuracies=[0],[initial_loss],[initial_accuracy]
    updates=100
    for step in range(updates):
        optimizer.zero_grad();loss,_=measure(train);loss.backward();optimizer.step()
        if (step+1)%10==0:
            with torch.no_grad():
                loss,accuracy=measure(train);steps.append(step+1);losses.append(float(loss));accuracies.append(accuracy)
    with torch.no_grad():
        final_loss,accuracy=measure(train);test_loss,test_accuracy=measure(held_out)
    return {'provenance':'PaperLeet synthetic teaching adaptation of GraphSAGE Algorithm1. Query-node attributes are zero; labels depend on sampled anchor neighborhoods. Shared encoder/classifier weights are trained on one 14-node graph, then frozen for a distinct permuted 18-node graph. Fixed sampled tables, no learned node-ID embeddings, no original dataset or paper metric reproduction.',
            'charts':[{'kind':'line','title':'Actual shared-weight optimization','x':steps,'x_label':'Adam update','y_label':'training cross-entropy','series':[{'name':'learner encoder','values':losses}]},
                      {'kind':'line','title':'Training query-node accuracy','x':steps,'x_label':'Adam update','y_label':'accuracy','series':[{'name':'zero-feature queries','values':accuracies}]},
                      {'kind':'bar','title':'Frozen model on distinct graphs','x':['training graph','unseen graph'],'series':[{'name':'query accuracy','values':[accuracy,test_accuracy]}]}],
            'metrics':{'updates':updates,'train_nodes':len(train[0]),'held_out_nodes':len(held_out[0]),'train_queries':len(train[3]),'held_out_queries':len(held_out[3]),'trainable_parameters':sum(value.numel() for value in [*weights,classifier,bias]),'initial_loss':initial_loss,'final_loss':float(final_loss),'initial_accuracy':initial_accuracy,'final_accuracy':accuracy,'held_out_loss':float(test_loss),'held_out_accuracy':test_accuracy,'feature_only_accuracy_ceiling':.5}}
