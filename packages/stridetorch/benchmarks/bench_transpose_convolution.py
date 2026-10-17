"""Transposed convolution versus zero insertion + existing regular convolution.

All comparisons are finite, same-source CPU/Pyodide flows. Controls force direct
and matrix routes; memory safety may still reject optional packed workspaces.
"""
import json
import math
import statistics
import time
import numpy as np
from stridetorch import torch,_transpose_convolution as tc


def composed(input,weight,bias,stride=2,padding=1,output_padding=1,groups=1,dilation=1):
    dim=weight.ndim-2;x=input
    for axis in range(2,dim+2):
        desired=(x.shape[axis]-1)*stride+1+output_padding
        if stride>1:
            x=torch.stack([x,*[torch.zeros_like(x) for _ in range(stride-1)]],dim=axis+1).flatten(axis,axis+1)
        if desired>x.shape[axis]:
            shape=list(x.shape);shape[axis]=desired-x.shape[axis]
            x=torch.cat([x,torch.zeros(shape,dtype=x.dtype)],dim=axis)
        elif desired<x.shape[axis]:
            key=[slice(None)]*x.ndim;key[axis]=slice(0,desired);x=x[tuple(key)]
    k=weight.shape[2:];cin=weight.shape[0];cout=weight.shape[1]*groups
    w=weight.reshape(groups,cin//groups,cout//groups,*k).transpose(1,2).reshape(cout,cin//groups,*k).flip(tuple(range(2,dim+2)))
    p=tuple(dilation*(v-1)-padding for v in k)
    return getattr(torch.nn.functional,f'conv{dim}d')(x,w,bias,padding=p,dilation=dilation,groups=groups)


def direct(input,weight,bias,**kwargs):
    return getattr(torch.nn.functional,f'conv_transpose{weight.ndim-2}d')(input,weight,bias,**kwargs)


def run(samples=5,calls=12,reverse_order=False):
    rng=np.random.default_rng(4816);results=[]
    cases=[('sequence',(2,4,16),6,3,1),('image',(2,4,8,8),8,3,1),('grouped',(2,4,8,8),8,3,2),('volume',(1,2,4,4,4),4,3,1),('larger_image',(2,8,16,16),8,3,1),('point',(2,4,8,8),6,1,1)]
    for kind,shape,cout,k,groups in cases:
        dim=len(shape)-2;options=dict(stride=2,padding=0 if k==1 else 1,output_padding=1,groups=groups)
        for dtype in [torch.float32,torch.float64]:
            arrays=[rng.normal(size=shape).astype(dtype),rng.normal(size=(shape[1],cout//groups,*([k]*dim))).astype(dtype),rng.normal(size=cout).astype(dtype)]
            directions=[rng.normal(size=a.shape).astype(dtype) for a in arrays]
            functions={'composed':lambda x,w,b:composed(x,w,b,**options),'auto':lambda x,w,b:direct(x,w,b,**options),'direct':lambda x,w,b:direct(x,w,b,**options),'matrix':lambda x,w,b:direct(x,w,b,**options)}
            seed=None
            def check(name):
                tc._ALGORITHM='auto' if name=='composed' else name
                values=tuple(torch.tensor(a,requires_grad=True) for a in arrays);ds=tuple(torch.tensor(a,requires_grad=True) for a in directions)
                p,t=torch.func.jvp(functions[name],values,ds)
                seed=torch.tensor(np.sin(np.arange(math.prod(p.shape))).reshape(p.shape).astype(dtype))
                (p*seed+t*seed.square()).sum().backward()
                return [v.numpy().copy() for v in (p,t,*[z.grad for z in (*values,*ds)])]
            qualification={name:dict(selected=0,fallback=0) for name in functions}
            original_selector=tc._use_matrix
            def checked(name):
                def select(*args):
                    result=original_selector(*args);qualification[name]['selected' if result else 'fallback']+=1;return result
                tc._use_matrix=select
                try:return check(name)
                finally:tc._use_matrix=original_selector
            expected=checked('composed')
            for name in ['auto','direct','matrix']:
                for actual,reference in zip(checked(name),expected):
                    np.testing.assert_allclose(actual,reference,rtol=5e-4 if dtype==torch.float32 else 8e-11,atol=5e-4 if dtype==torch.float32 else 8e-11)
            seed=torch.tensor(np.sin(np.arange(expected[0].size)).reshape(expected[0].shape).astype(dtype))
            for mode in ['forward','reverse','jvp']:
                def action(name):
                    tc._ALGORITHM='auto' if name=='composed' else name
                    if mode=='forward':
                        with torch.no_grad():return functions[name](*[torch.tensor(a) for a in arrays])
                    values=tuple(torch.tensor(a,requires_grad=True) for a in arrays)
                    if mode=='reverse':y=functions[name](*values);y.backward(seed);return y
                    return torch.func.jvp(functions[name],values,tuple(torch.tensor(a) for a in directions))
                for name in functions:action(name)
                times={name:[] for name in functions}
                for sample in range(samples):
                    order=list(functions)
                    if (sample+int(reverse_order))%2:order.reverse()
                    for name in order:
                        start=time.perf_counter()
                        for _ in range(calls):action(name)
                        times[name].append((time.perf_counter()-start)/calls)
                results.append(dict(kind=kind,shape=list(shape),dtype=str(dtype),mode=mode,qualification=qualification,samples_s=times,median_s={k:statistics.median(v) for k,v in times.items()}))
    for dtype in [torch.float32,torch.float64]:
        arrays=[rng.normal(size=(2,4,3,3)).astype(dtype)*.1,np.ones(4,dtype=dtype),np.zeros(4,dtype=dtype)]
        x=torch.tensor(rng.normal(size=(2,2,6,6)).astype(dtype));target=torch.tensor(rng.normal(size=(2,4,12,12)).astype(dtype))
        def make(name):
            params=[torch.nn.Parameter(torch.tensor(a)) for a in arrays];optimizer=torch.optim.SGD(params,lr=.01,momentum=.7)
            def step():
                tc._ALGORITHM='auto' if name=='composed' else name
                optimizer.zero_grad();y=(composed if name=='composed' else direct)(x,params[0],None,stride=2,padding=1,output_padding=1)
                y=torch.nn.functional.instance_norm(y,weight=params[1],bias=params[2]);loss=(torch.nn.functional.gelu(y)-target).square().mean()
                loss.backward();optimizer.step();return loss,params
            return step
        baseline=make('composed');expected=[]
        for _ in range(3):
            loss,p=baseline();expected.append([loss.numpy().copy(),*[v.numpy().copy() for v in p]])
        for name in ['auto','direct','matrix']:
            step=make(name)
            for reference in expected:
                loss,p=step()
                for a,b in zip([loss,*p],reference):np.testing.assert_allclose(a.numpy(),b,rtol=5e-5 if dtype==torch.float32 else 8e-11,atol=5e-6 if dtype==torch.float32 else 8e-11)
        times={name:[] for name in ['composed','auto','direct','matrix']}
        for sample in range(samples):
            order=list(times)
            if (sample+int(reverse_order))%2:order.reverse()
            for name in order:
                step=make(name);step();start=time.perf_counter()
                for _ in range(max(3,calls//4)):step()
                times[name].append((time.perf_counter()-start)/max(3,calls//4))
        results.append(dict(kind='decoder',shape=[2,2,6,6],dtype=str(dtype),mode='decoder_sgd',samples_s=times,median_s={k:statistics.median(v) for k,v in times.items()}))
    tc._ALGORITHM='auto'
    return dict(status='success',baseline='same-source zero insertion + grouped flipped regular convolution',correctness='finite values, VJP/JVP and reverse-over-forward, three matching decoder SGD updates',flows=results,samples=samples,calls=calls)


if __name__=='__main__':print(json.dumps(run(),indent=2))
