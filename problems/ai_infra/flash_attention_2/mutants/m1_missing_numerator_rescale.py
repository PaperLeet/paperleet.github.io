# Bug: 最大值变化时没有重缩放旧分子
import math
import torch
from torch import Tensor

def flash2_forward(q: Tensor, k: Tensor, v: Tensor, query_block: int, key_block: int, causal: bool = False) -> tuple[Tensor,Tensor]:
    if any(type(b) is not int or b<=0 for b in (query_block,key_block)) or type(causal) is not bool:
        raise ValueError('positive block sizes and boolean causal flag required')
    if q.ndim!=2 or k.shape!=q.shape or v.ndim!=2 or v.shape[0]!=q.shape[0] or min(*q.shape,*v.shape)<=0:
        raise ValueError('matching nonempty square-attention inputs required')
    if any(t.dtype!=q.dtype or t.device!=q.device for t in (k,v)):
        raise ValueError('matching dtype and device required')
    outputs=[]; logs=[]; n=q.shape[0]
    for start in range(0,n,query_block):
        end=min(n,start+query_block); qi=q[start:end]
        m=None; ell=None; numerator=None
        for j in range(0,n,key_block):
            if causal and j>=end: break
            stop=min(n,j+key_block)
            scores=qi@k[j:stop].T/math.sqrt(q.shape[1])
            if causal:
                allowed=torch.arange(j,stop,device=q.device)[None,:]<=torch.arange(start,end,device=q.device)[:,None]
                scores=scores.masked_fill(~allowed,float('-inf'))
            block_max=scores.max(dim=1).values
            new_max=block_max if m is None else torch.maximum(m,block_max)
            weights=torch.exp(scores-new_max[:,None])
            if m is None:
                ell=weights.sum(dim=1); numerator=weights@v[j:stop]
            else:
                correction=torch.exp(m-new_max)
                ell=correction*ell+weights.sum(dim=1)
                numerator=numerator+weights@v[j:stop]
            m=new_max
        outputs.append(numerator/ell[:,None]); logs.append(m+torch.log(ell))
    return torch.cat(outputs,dim=0),torch.cat(logs,dim=0)
