"""A bounded table-and-linear-predictor experiment, with a real EMA after every SGD step."""
import torch
from .toy import fixture, scalar_loss


def run(byol_loss, ema_update):
    a, b, _, _ = fixture()
    a=a.clone().requires_grad_(); b=b.clone().requires_grad_()
    predictor=torch.tensor([[.2,.7,-.3],[-.4,.3,.8],[.6,-.2,.1]],dtype=a.dtype,requires_grad=True)
    ta,tb=a.detach().clone(),b.detach().clone()
    original_a,original_b=ta.clone(),tb.clone()
    optimizer=torch.optim.SGD([a,b,predictor],lr=.18)
    steps,losses,lags=[],[],[]; scalar_error=0.; ema_error=0.
    for step in range(81):
        p1,p2=a@predictor,b@predictor
        loss=byol_loss(p1,p2,ta,tb)
        oracle=scalar_loss(*(v.detach().tolist() for v in (p1,p2,ta,tb)))
        scalar_error=max(scalar_error,abs(float(loss.detach())-oracle))
        if step%8==0:
            steps.append(step);losses.append(float(loss.detach()));lags.append(float(((ta-a.detach()).square().mean()+(tb-b.detach()).square().mean())/2))
        if step<80:
            optimizer.zero_grad();loss.backward();optimizer.step()
            next_a,next_b=ema_update(a,ta,.9),ema_update(b,tb,.9)
            for online,old,new in ((a,ta,next_a),(b,tb,next_b)):
                wanted=[[.9*x+.1*y for x,y in zip(ro,rn)] for ro,rn in zip(old.tolist(),online.detach().tolist())]
                ema_error=max(ema_error,max(abs(x-y) for row,ref in zip(new.tolist(),wanted) for x,y in zip(row,ref)))
            ta,tb=next_a,next_b
    return {'kind':'toy_experiment','title':'BYOL: prediction and moving targets','problem':'representation/byol',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2006.07733',
            'anchor':'Section 3.1 and Algorithm 1','adaptation':'Two 6x3 online representation tables and a shared 3x3 linear predictor learn with 80 real SGD updates and parameter-table EMA. These independent tables isolate bootstrapped targets; no image encoder, augmentations, downstream evaluation or no-collapse guarantee.'},
        'metrics':{'updates':80,'online_parameters':45,'initial_loss':losses[0],'final_loss':losses[-1],
            'scalar_error':scalar_error,'ema_error':ema_error,'target_movement':float((ta-original_a).abs().sum()+(tb-original_b).abs().sum()),
            'targets_detached':not ta.requires_grad and not tb.requires_grad},
        'charts':[{'kind':'line','title':'Submitted BYOL regression','x':steps,'x_label':'SGD update','series':[{'name':'Loss','values':losses}]},
            {'kind':'line','title':'Target lag behind online tables','x':steps,'x_label':'SGD update','series':[{'name':'Mean squared lag','values':lags}]}]}
