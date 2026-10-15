from paperleet.common.asserts import assert_close_diagnostic as close
import math
import torch


def run(solve):
    margins=[-4.,-2.,0.,2.,4.]; values=[]
    tokens=torch.tensor([[2,0,1,2],[0,2,0,1]])
    for margin in margins:
        logits=torch.zeros(2,4,3,dtype=torch.float64)
        for b in range(2):
            for t in range(3): logits[b,t,int(tokens[b,t+1])]=margin
        logits[:,-1,:]=torch.tensor([1000.,-1000.,0.],dtype=logits.dtype)
        loss=solve(logits,tokens)
        close(loss,torch.tensor(math.log1p(2*math.exp(-margin)),dtype=logits.dtype))
        values.append(loss.item())
    assert values[-1]<values[0]
    return {'provenance':'Original deterministic next-token teaching experiment; no model benchmark.',
            'charts':[{'kind':'line','title':'Next-token confidence','x':margins,'x_label':'correct next-token logit margin','y_label':'mean NLL','series':[{'name':'Learner loss','values':values}]}],
            'metrics':{'low_confidence_loss':values[0],'high_confidence_loss':values[-1]}}
