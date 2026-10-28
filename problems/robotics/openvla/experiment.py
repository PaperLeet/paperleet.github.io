"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='OpenVLA'
IDENTIFIER='robotics/openvla'
SOURCE='https://arxiv.org/abs/2406.09246'
ANCHOR='Official action_tokenizer.py digitize / reversed IDs; OpenVLA.predict_action quantile unnormalization'
ADAPTATION='Reproduce official ActionTokenizer conventions and action unnormalization without a 7B model. Smaller edge arrays make examples inspectable; the official default has 256 edges and 255 centers.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(encode_actions,decode_actions,unnormalize_actions):
    e=torch.linspace(-1,1,33,dtype=torch.float64);x=torch.linspace(-1.2,1.2,99,dtype=e.dtype).reshape(33,3)
    tokens=encode_actions(x,e,1000);decoded=decode_actions(tokens,e,1000);lo=e.new_tensor([-2.,1.,0.]);hi=e.new_tensor([2.,5.,2.]);mask=torch.tensor([True,True,False])
    physical=unnormalize_actions(decoded,lo,hi,mask);target=torch.where(mask,.5*(x.clamp(-1,1)+1)*(hi-lo)+lo,x.clamp(-1,1));error=(physical-target).abs()
    return _report({'max_normalized_error':(decoded-x.clamp(-1,1)).abs().max().item(),'max_physical_error':error.max().item(),'roundtrip_codes':int((encode_actions(decoded,e,1000)==tokens).sum().item())},error.mean(dim=1).tolist(),'Quantization error per action')
