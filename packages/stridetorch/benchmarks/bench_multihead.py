"""Packed MHA vs the same library's three-projection functional control.

This isolates projection packing, not a comparison with a previous release or
native PyTorch. Each flow checks values/derivatives before measuring both paths.
"""
import json
import statistics
import time
import numpy as np

from stridetorch import torch


def run(samples=5, calls=4):
    torch.manual_seed(5208)
    rng=np.random.default_rng(5208);rows=[]
    for dtype in ['float32','float64']:
        dt=getattr(torch,dtype)
        for label,batch,length,embed,heads in [('small',2,8,16,4),('encoder',2,64,64,4),('long',1,128,64,4)]:
            options=dict(dtype=dt)
            m=torch.nn.MultiheadAttention(embed,heads,batch_first=True,**options)
            norm=torch.nn.LayerNorm(embed,**options)
            ff=torch.nn.Sequential(torch.nn.Linear(embed,embed*2,**options),torch.nn.GELU(),torch.nn.Linear(embed*2,embed,**options))
            parameters=list(m.parameters())+list(norm.parameters())+list(ff.parameters())
            arrays=[rng.normal(0,.2,size=(batch,length,embed)).astype(dtype) for _ in range(3)]
            def attention(x,packed):
                if packed:return m(x,x,x,need_weights=False)[0]
                q=x.transpose(0,1)
                y,_=torch.nn.functional.multi_head_attention_forward(q,q,q,embed,heads,
                    None,m.in_proj_bias,None,None,False,0.,m.out_proj.weight,m.out_proj.bias,
                    need_weights=False,use_separate_proj_weight=True,q_proj_weight=m.in_proj_weight[:embed],
                    k_proj_weight=m.in_proj_weight[embed:2*embed],v_proj_weight=m.in_proj_weight[2*embed:])
                return y.transpose(0,1)
            def encoder(x,packed):
                y=x+attention(norm(x),packed)
                return y+ff(y)
            for packed in [True,False]:
                x=torch.tensor(arrays[0],requires_grad=True)
                y=attention(x,packed);grads=torch.autograd.grad(y.square().sum(),[x,*m.parameters()])
                p,t=torch.func.jvp(lambda a:attention(a,packed),(x,),(torch.tensor(arrays[1]),))
                evidence=[z.numpy().copy() for z in (y,*grads,p,t)]
                if packed:expected=evidence
                else:
                    for a,b in zip(evidence,expected):np.testing.assert_allclose(a,b,rtol=5e-5 if dtype=='float32' else 1e-11,atol=5e-6 if dtype=='float32' else 1e-12)
            initial=[p.detach().numpy().copy() for p in parameters]
            def restore():
                with torch.no_grad():
                    for p,a in zip(parameters,initial):p.copy_(torch.tensor(a))
            def training(packed,verify=False):
                o=torch.optim.SGD(parameters,lr=.005,momentum=.7)
                for _ in range(4):
                    o.zero_grad();x=torch.tensor(arrays[0]);encoder(x,packed).square().mean().backward();o.step()
                if verify:return [p.detach().numpy().copy() for p in parameters]
            restore();a=training(True,verify=True)
            restore();b=training(False,verify=True)
            for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=5e-5 if dtype=='float32' else 1e-11,atol=5e-6 if dtype=='float32' else 1e-12)
            restore()
            def invoke(mode,packed):
                if mode=='training4':return training(packed)
                x=torch.tensor(arrays[0],requires_grad=mode=='reverse')
                if mode=='forward':
                    with torch.no_grad():attention(x,packed)
                elif mode=='reverse':torch.autograd.grad(attention(x,packed).square().sum(),[x,*m.parameters()])
                else:torch.func.jvp(lambda a:attention(a,packed),(x,),(torch.tensor(arrays[1]),))
            for mode in ['forward','reverse','jvp','training4']:
                for packed in [True,False]:invoke(mode,packed)
                times=dict(packed=[],three_projection_control=[])
                repeat=1 if mode=='training4' else calls
                for sample in range(samples):
                    for packed in ([True,False] if sample%2==0 else [False,True]):
                        if mode=='training4':restore()
                        start=time.perf_counter()
                        for _ in range(repeat):invoke(mode,packed)
                        times['packed' if packed else 'three_projection_control'].append((time.perf_counter()-start)/repeat)
                medians={k:statistics.median(v) for k,v in times.items()}
                rows.append(dict(case=label,dtype=dtype,mode=mode,samples_s=times,median_s=medians,
                                 packed_over_control=medians['packed']/medians['three_projection_control']))
    return dict(scope='Same-library packed self-attention versus three independent Q/K/V projections; no native Torch or previous-version timing claim',samples=samples,calls=calls,rows=rows)


if __name__=='__main__':print(json.dumps(run(),indent=2))
