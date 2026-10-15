"""Twelve actual SGD/EMA/dictionary steps checked against an independent scalar trajectory."""
import math
import torch
from .toy import fixture, scalar_step


def run(moco_loss, momentum_update, enqueue):
    xq,xk,initial,queue=fixture();query=torch.nn.Parameter(initial.clone());key=initial.clone();pointer=0
    expected_q,expected_k,expected_queue=initial.tolist(),initial.tolist(),queue.tolist();expected_ptr=0
    optimizer=torch.optim.SGD([query],lr=.05)
    steps=[];losses=[];lags=[];pointers=[];maximum_loss_error=maximum_state_error=maximum_gradient=0.
    def error(a,b):return max((abs(x-y) for row,other in zip(a.tolist(),b) for x,y in zip(row,other)),default=0.)
    for step in range(12):
        indices=torch.tensor([(2*step)%6,(2*step+1)%6]);a,b=xq[indices],xk[indices]
        expected_loss,expected_q,expected_k,expected_queue,expected_ptr=scalar_step(expected_q,expected_k,expected_queue,expected_ptr,a.tolist(),b.tolist(),.3,.05,.8)
        q,k=a@query,b@key
        loss=moco_loss(q,k,queue,.3);optimizer.zero_grad();loss.backward()
        maximum_gradient=max(maximum_gradient,float(query.grad.abs().max()));optimizer.step()
        key=momentum_update(query,key,.8);queue,pointer=enqueue(queue,k,pointer)
        maximum_loss_error=max(maximum_loss_error,abs(float(loss.detach())-expected_loss))
        maximum_state_error=max(maximum_state_error,error(query,expected_q),error(key,expected_k),error(queue,expected_queue),abs(pointer-expected_ptr))
        steps.append(step+1);losses.append(float(loss.detach()));lags.append(float((query.detach()-key).square().sum().sqrt()));pointers.append(pointer)
    provenance='Original synthetic CPU adaptation: six fixed 4D view pairs, 4x3 linear query encoder (12 trainable scalars), detached EMA key encoder, five queued keys, twelve actual SGD updates (lr .05, m .8, temperature .3). State trajectory checked against scalar finite differences. No image encoder, augmentation pipeline, shuffling BN, distributed training or downstream benchmark.'
    return {'provenance':provenance,'charts':[
        {'kind':'line','title':'Actual per-batch InfoNCE loss','x':steps,'x_label':'SGD update','series':[{'name':'Training batch loss','values':losses}]},
        {'kind':'line','title':'Momentum encoder lag','x':steps,'x_label':'SGD update','y_label':'L2 parameter distance','series':[{'name':'Query minus key','values':lags}]},
        {'kind':'line','title':'Circular queue pointer','x':steps,'x_label':'SGD update','series':[{'name':'Next overwritten slot','values':pointers}]}],
        'metrics':{'updates':12,'trainable_parameters':12,'dictionary_size':5,'maximum_scalar_loss_error':maximum_loss_error,'maximum_state_error':maximum_state_error,'maximum_query_gradient':maximum_gradient,'final_pointer':pointer,'final_loss':losses[-1]}}
