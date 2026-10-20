"""Comparable float arithmetic costs, with values and gradients checked first."""
import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np
from torchwasm import torch, __version__


def run_benchmarks(repeat=11, iterations=120, warmup=3, dtype='float32', cases=None):
    kind=np.dtype(dtype)
    if dtype not in ('float32','float64') or min(repeat,iterations)<1 or warmup<0:raise ValueError('Invalid arithmetic benchmark options')
    raw=np.full((64,128),.25,dtype=kind);rhs=np.full((128,),.75,dtype=kind)
    a,b=torch.tensor(raw),torch.tensor(rhs)
    x,y=torch.tensor(raw,requires_grad=True),torch.tensor(rhs,requires_grad=True)
    forward={
        'add_tensor':lambda:a+b,'add_scalar':lambda:a+.125,
        'sub_tensor':lambda:a-b,'reverse_sub_scalar':lambda:1.5-a,
        'mul_tensor':lambda:a*b,'mul_scalar':lambda:a*.5,
        'div_tensor':lambda:a/b,'div_scalar':lambda:a/1.5,
        'reverse_div_scalar':lambda:1.5/a,
    }
    expected={'add_tensor':raw+rhs,'add_scalar':raw+.125,'sub_tensor':raw-rhs,
              'reverse_sub_scalar':1.5-raw,'mul_tensor':raw*rhs,'mul_scalar':raw*.5,
              'div_tensor':raw/rhs,'div_scalar':raw/1.5,'reverse_div_scalar':1.5/raw}
    def backward(name):
        x.grad=y.grad=None
        value=x+y if name=='add_backward' else x*y if name=='mul_backward' else x/y
        loss=value.square().mean();loss.backward();return loss
    def chain():
        x.grad=None;value=x
        for _ in range(16):value=value*.5+.125
        loss=value.square().mean();loss.backward();return loss
    available={**forward,**{name:(lambda name=name:backward(name)) for name in ['add_backward','mul_backward','div_backward']},'affine_chain_backward':chain}
    selected=list(available) if cases is None else list(cases)
    if not selected or len(set(selected))!=len(selected) or any(x not in available for x in selected):raise ValueError('Choose unique arithmetic cases')
    reports,contracts={},{}
    for name in selected:
        run=available[name];value=run();assert value.dtype==kind,(name,value.dtype)
        contracts[name]={'shape':list(value.shape),'dtype':str(value.dtype)}
        if name in expected:np.testing.assert_allclose(value.numpy(),expected[name],rtol=1e-6)
        elif name=='affine_chain_backward':
            np.testing.assert_allclose(value.item(),.0625,rtol=1e-6)
            np.testing.assert_allclose(x.grad.numpy(),np.full_like(raw,2*.25/raw.size*.5**16),rtol=1e-6,atol=1e-15)
        else:
            result=raw+rhs if name=='add_backward' else raw*rhs if name=='mul_backward' else raw/rhs
            seed=result*np.asarray(2/raw.size,dtype=kind)
            dx=seed if name=='add_backward' else seed*rhs if name=='mul_backward' else seed/rhs
            dy=seed if name=='add_backward' else seed*raw if name=='mul_backward' else -seed*raw/rhs**2
            np.testing.assert_allclose(x.grad.numpy(),dx,rtol=1e-6);np.testing.assert_allclose(y.grad.numpy(),dy.sum(0),rtol=1e-6)
        del value
        inner=iterations*10 if name in forward else max(1,iterations//4) if name=='affine_chain_backward' else iterations
        for _ in range(warmup):run()
        samples=[]
        for _ in range(repeat):
            start=time.perf_counter()
            for _ in range(inner):run()
            samples.append((time.perf_counter()-start)*1000/inner)
        reports[name]={'median_ms':statistics.median(samples),'samples_ms':samples,'iterations':inner}
    return dict(library_version=__version__,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                implementation=sys.modules[torch.Tensor.__module__].__file__,dtype=dtype,shape=[64,128],chain_depth=16,
                repeat=repeat,iterations=iterations,warmup=warmup,contracts=contracts,results=reports)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat',type=int,default=11);parser.add_argument('--iterations',type=int,default=120)
    parser.add_argument('--dtype',choices=['float32','float64'],default='float32');parser.add_argument('--case',action='append',dest='cases');parser.add_argument('--output')
    args=parser.parse_args();raw=json.dumps(run_benchmarks(args.repeat,args.iterations,dtype=args.dtype,cases=args.cases),indent=2)+'\n'
    if args.output:
        with open(args.output,'w',encoding='utf8') as stream:stream.write(raw)
    print(raw)
