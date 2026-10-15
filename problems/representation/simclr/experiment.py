"""Optimize two tiny synthetic tables; no encoder, augmentation, dataset or paper benchmark."""
import torch
from .toy import fixture, measure


def run(nt_xent_loss):
    a,b=fixture()
    a=torch.nn.Parameter(a);b=torch.nn.Parameter(b)
    optimizer=torch.optim.Adam([a,b],lr=.07)
    temperature,updates=.2,120
    steps=[];losses=[];positives=[];negatives=[];accuracies=[];maximum_error=0.
    def checkpoint(step):
        nonlocal maximum_error
        with torch.no_grad():
            metrics=measure(a,b,temperature)
            actual=float(nt_xent_loss(a,b,temperature))
        maximum_error=max(maximum_error,abs(actual-metrics['loss']))
        steps.append(step);losses.append(actual);positives.append(metrics['positive'])
        negatives.append(metrics['negative']);accuracies.append(metrics['accuracy'])
        return metrics
    initial=checkpoint(0)
    for step in range(1,updates+1):
        optimizer.zero_grad()
        loss=nt_xent_loss(a,b,temperature);loss.backward();optimizer.step()
        if step%10==0: final=checkpoint(step)
    provenance='Original synthetic CPU adaptation: 6 paired instances, two independently initialized 6x8 tables (96 trainable scalars), 120 full-batch Adam updates, temperature .2. Training metrics only; no images, encoder, augmentation or downstream evaluation.'
    return {'provenance':provenance,
        'charts':[
            {'kind':'line','title':'Actual NT-Xent training loss','x':steps,'x_label':'Adam update','y_label':'mean over 12 anchors','series':[{'name':'Training loss','values':losses}]},
            {'kind':'line','title':'Paired similarity versus hardest negative','x':steps,'x_label':'Adam update','y_label':'mean cosine similarity','series':[{'name':'Positive','values':positives},{'name':'Hardest negative','values':negatives}]},
            {'kind':'line','title':'Training paired retrieval','x':steps,'x_label':'Adam update','y_label':'top-1 fraction excluding self','series':[{'name':'Training accuracy','values':accuracies}]},
            {'kind':'bar','title':'Final per-anchor separation','x':[f'A{i+1}' for i in range(6)]+[f'B{i+1}' for i in range(6)],'x_label':'Anchor','series':[{'name':'Positive','values':final['positive_per_anchor']},{'name':'Hardest negative','values':final['hardest_negative_per_anchor']}]}],
        'metrics':{'updates':updates,'trainable_parameters':a.numel()+b.numel(),'pairs':len(a),'temperature':temperature,
                   'initial_loss':losses[0],'final_loss':losses[-1],'initial_training_accuracy':initial['accuracy'],
                   'final_training_accuracy':final['accuracy'],'final_minimum_margin':final['minimum_margin'],
                   'maximum_scalar_error':maximum_error}}
