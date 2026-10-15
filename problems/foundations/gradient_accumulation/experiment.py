from paperleet.common.asserts import assert_close_diagnostic as close
import torch


def run(solve):
    x=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,-1.],[-1.,2.],[.5,1.]],dtype=torch.float64)
    y=x@torch.tensor([1.5,-.75],dtype=x.dtype)
    w=torch.zeros(2,dtype=x.dtype);full=w.clone();losses=[];reference=[];gaps=[]
    for step in range(21):
        losses.append(float((x@w-y).square().mean()));reference.append(float((x@full-y).square().mean()))
        gaps.append(float((w-full).abs().max()))
        if step==20:break
        grads=[]
        for start,end in [(0,1),(1,6)]:
            xx=x[start:end];residual=xx@w-y[start:end]
            grads.append(2*xx.T@residual/(end-start))
        effective=solve(grads,[1,5]);w=w-.08*effective
        full=full-.08*(2*x.T@(x@full-y)/6)
    close(w,full,atol=1e-10,rtol=1e-10)
    assert losses[-1]<losses[0]*.05
    return {'provenance':'Original synthetic linear-model training with unequal microbatches; no full-LLM benchmark.',
            'charts':[{'kind':'line','title':'Effective-batch training','x':list(range(21)),'x_label':'update','y_label':'MSE','series':[{'name':'Learner accumulation','values':losses},{'name':'Full-batch oracle','values':reference}]}],
            'metrics':{'initial_loss':losses[0],'final_loss':losses[-1],'maximum_parameter_gap':max(gaps)}}
