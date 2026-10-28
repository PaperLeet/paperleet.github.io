# Bug: 误差补偿使用相反符号
import torch
from torch import Tensor

def gptq_quantize(weight: Tensor, upper: Tensor, scales: Tensor, qmin: int = -7, qmax: int = 7) -> Tensor:
    if weight.ndim!=2 or min(weight.shape)<=0 or upper.shape!=(weight.shape[1],weight.shape[1]) or scales.shape!=(weight.shape[0],):
        raise ValueError('incompatible weights, factor and per-row scales')
    if type(qmin) is not int or type(qmax) is not int or qmin>=qmax or bool((scales<=0).any()):
        raise ValueError('positive scales and ordered integer bounds required')
    if any(t.dtype!=weight.dtype or t.device!=weight.device for t in (upper,scales)):
        raise ValueError('matching dtype and device required')
    for i in range(upper.shape[0]):
        if float(upper[i,i])<=0 or bool((upper[i,:i]!=0).any()): raise ValueError('upper Cholesky factor required')
    working=weight.detach().clone(); u=upper.detach(); scale=scales.detach(); columns=[]
    for j in range(weight.shape[1]):
        current=working[:,j].clone()
        quantized=torch.round(current/scale).clamp(qmin,qmax)*scale
        columns.append(quantized)
        error=(current-quantized)/u[j,j]
        working[:,j:]=working[:,j:]+error[:,None]*u[j,j:][None,:]
    return torch.stack(columns,dim=1)
