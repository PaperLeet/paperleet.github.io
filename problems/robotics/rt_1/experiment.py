"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='RT-1'
IDENTIFIER='robotics/rt_1'
SOURCE='https://arxiv.org/abs/2212.06817'
ANCHOR='Official tokenizers/token_learner.py: spatial softmax and einsum over original features'
ADAPTATION='Isolate visual-token compression with supplied MLP attention logits; omit EfficientNet, the Transformer and physical robots.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(token_learner_pool):
    x=torch.tensor([[[-1.],[0.],[1.]]],dtype=torch.float64);scores=torch.zeros(1,3,1,dtype=x.dtype,requires_grad=True);curve=[]
    for i in range(101):
        pooled=token_learner_pool(x,scores);loss=(pooled-.8).square().mean();curve.append(loss.item())
        if i<100:g,=torch.autograd.grad(loss,scores);scores=(scores-g).detach().requires_grad_()
    return _report({'initial_loss':curve[0],'final_loss':curve[-1],'pooled':pooled.item()},curve,'Patch selection loss')
