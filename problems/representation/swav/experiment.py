"""Balance fixed prototype evidence, then learn its swapped assignments in two score tables."""
import math
import torch
from .toy import fixture,scalar_loss,scalar_assignments


def run(sinkhorn_assignments,swapped_prediction_loss):
    raw1,raw2=fixture();bias=torch.tensor([[.4,-.1,-.2]],dtype=raw1.dtype)
    evidence1,evidence2=raw1+bias,raw2+bias
    q1=sinkhorn_assignments(evidence1,.2,40);q2=sinkhorn_assignments(evidence2,.2,40)
    oracle_q1=scalar_assignments(evidence1.tolist(),.2,40);oracle_q2=scalar_assignments(evidence2.tolist(),.2,40)
    assignment_error=max(abs(x-y) for q,r in ((q1.tolist(),oracle_q1),(q2.tolist(),oracle_q2)) for row,ref in zip(q,r) for x,y in zip(row,ref))
    score1=(-raw2).clone().requires_grad_();score2=(-raw1).clone().requires_grad_()
    optimizer=torch.optim.Adam([score1,score2],lr=.04)
    entropy=-sum(q*math.log(q) for table in (oracle_q1,oracle_q2) for row in table for q in row if q>0)/12
    steps,losses=[],[];scalar_error=0.
    for step in range(81):
        loss=swapped_prediction_loss(score1,score2,q1,q2,.3)
        expected=scalar_loss(score1.detach().tolist(),score2.detach().tolist(),oracle_q1,oracle_q2,.3)
        scalar_error=max(scalar_error,abs(float(loss.detach())-expected))
        if step%8==0:steps.append(step);losses.append(float(loss.detach()))
        if step<80:optimizer.zero_grad();loss.backward();optimizer.step()
    return {'kind':'toy_experiment','title':'SwAV: balance assignments, swap predictions','problem':'representation/swav',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2006.09882','anchor':'Section 3, online clustering and swapped prediction',
            'adaptation':'40 single-process Sinkhorn iterations on two fixed 6x3 evidence tables, followed by 80 Adam updates of 36 prediction scores. This isolates balanced soft targets and swapped prediction; no image encoder, prototype learning, queue or multi-crop benchmark.'},
        'metrics':{'updates':80,'trainable_parameters':36,'initial_loss':losses[0],'final_loss':losses[-1],'assignment_entropy':entropy,
            'final_kl':losses[-1]-entropy,'scalar_error':scalar_error,'assignment_error':assignment_error,
            'maximum_balance_error':max(float((q.sum(0)-2).abs().max()) for q in (q1,q2)),
            'assignments_detached':not q1.requires_grad and not q2.requires_grad},
        'charts':[{'kind':'line','title':'Submitted swapped prediction objective','x':steps,'x_label':'Adam update',
            'series':[{'name':'Cross-entropy','values':losses},{'name':'Soft-target entropy','values':[entropy]*len(steps)}]},
            {'kind':'bar','title':'Balanced assignment mass per prototype','x':['P1','P2','P3'],
            'series':[{'name':'First view','values':q1.sum(0).tolist()},{'name':'Second view','values':q2.sum(0).tolist()}]}]}
