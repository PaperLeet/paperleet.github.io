from paperleet.common.asserts import assert_close_diagnostic as close
import math
import torch


def run(solve):
    spikes=[.1,.5,1.,5.,20.];raw=[];clipped=[];updates=[]
    for spike in spikes:
        grads=[torch.tensor([3.*spike],dtype=torch.float64),torch.tensor([4.*spike],dtype=torch.float64)]
        out,norm=solve(grads,2.)
        expected=5*spike;scale=min(1.,2/(expected+1e-6))
        close(norm,torch.tensor(expected,dtype=norm.dtype))
        for actual,g in zip(out,grads):close(actual,g*scale)
        measured=math.sqrt(sum(float(v.square().sum()) for v in out))
        assert measured<=2.+1e-12
        raw.append(norm.item());clipped.append(measured);updates.append(.1*measured)
    return {'provenance':'Original synthetic cross-parameter gradient-spike experiment; no model convergence claim.',
            'charts':[{'kind':'line','title':'One global clipping scale','x':spikes,'x_label':'gradient multiplier','y_label':'global L2 norm','series':[{'name':'Measured original norm','values':raw},{'name':'Learner clipped norm','values':clipped}]}],
            'metrics':{'largest_raw_norm':max(raw),'largest_clipped_norm':max(clipped),'largest_SGD_update_at_lr_0_1':max(updates)}}
