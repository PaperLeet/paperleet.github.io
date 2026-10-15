import math
import torch


def run(solve):
    margins=[-8.,-4.,0.,4.,8.];values=[]
    tokens=torch.ones(2,4,dtype=torch.long)
    response=torch.tensor([[False,False,True,True],[False,False,True,True]])
    attention=torch.tensor([[True,True,True,True],[True,True,True,False]])
    expected=(2*math.log1p(math.exp(-1))+math.log1p(math.exp(-2)))/3
    for margin in margins:
        x=torch.zeros(2,4,2,dtype=torch.float64)
        x[:,0,1]=margin; x[:,1,1]=1.;x[:,2,1]=2.;x[:,-1,0]=1000.
        loss=solve(x,tokens,response,attention)
        torch.testing.assert_close(loss,torch.tensor(expected,dtype=x.dtype))
        values.append(loss.item())
    return {'provenance':'Original synthetic prompt/response masking experiment; no model benchmark.',
            'charts':[{'kind':'line','title':'Exclude prompt predictions','x':margins,'x_label':'ignored prompt logit margin','y_label':'response-only NLL','series':[{'name':'Learner masked loss','values':values}]}],
            'metrics':{'valid_response_tokens':3,'response_loss':values[0],'loss_variation':max(values)-min(values)}}
