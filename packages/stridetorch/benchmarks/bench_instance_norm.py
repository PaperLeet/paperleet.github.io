"""InstanceNorm versus an equivalent pre-existing BatchNorm composition.

Run with Python or mount this file and stridetorch in a Pyodide worker.
Timings include Python dispatch and tape construction; they are not kernel-only.
"""
import json
import math
import statistics
import time
import numpy as np
from stridetorch import torch


def composed(input, weight, bias):
    n,c = input.shape[:2]
    folded = input.contiguous().reshape(1,n*c,*input.shape[2:])
    return torch.nn.functional.batch_norm(folded,None,None,
        weight.repeat(n),bias.repeat(n),training=True).reshape(input.shape)


def direct(input,weight,bias):
    return torch.nn.functional.instance_norm(input,weight=weight,bias=bias)


def run(samples=5,calls=12,reverse_order=False):
    rng = np.random.default_rng(4714)
    results = []
    for shape in [(4,16,32),(2,8,16,16),(2,4,8,8,8),(4,32,32,32)]:
        for dtype in [torch.float32,torch.float64]:
            arrays = [rng.normal(size=shape).astype(dtype),rng.normal(size=shape[1]).astype(dtype),rng.normal(size=shape[1]).astype(dtype)]
            directions = [rng.normal(size=a.shape).astype(dtype) for a in arrays]
            seed = torch.tensor(rng.normal(size=shape).astype(dtype))
            def check(fn):
                values = tuple(torch.tensor(a,requires_grad=True) for a in arrays)
                tangent_inputs = tuple(torch.tensor(a,requires_grad=True) for a in directions)
                y,t = torch.func.jvp(fn,values,tangent_inputs)
                (y*seed+t*seed.square()).sum().backward()
                return [v.detach().numpy().copy() for v in (y,t,*[a.grad for a in (*values,*tangent_inputs)])]
            expected = check(composed)
            for actual,reference in zip(check(direct),expected):
                np.testing.assert_allclose(actual,reference,rtol=3e-4 if dtype==torch.float32 else 3e-11,atol=3e-5 if dtype==torch.float32 else 3e-11)
            for mode in ['forward','reverse','jvp']:
                def action(fn):
                    if mode == 'forward':
                        with torch.no_grad(): return fn(*[torch.tensor(a) for a in arrays])
                    values = tuple(torch.tensor(a,requires_grad=True) for a in arrays)
                    if mode == 'reverse':
                        value = fn(*values);value.backward(seed);return value
                    return torch.func.jvp(fn,values,tuple(torch.tensor(a) for a in directions))
                for fn in [composed,direct]: action(fn)
                timings = {'composed':[],'direct':[]}
                for sample in range(samples):
                    order = [('composed',composed),('direct',direct)]
                    if (sample+int(reverse_order))%2: order.reverse()
                    for name,fn in order:
                        start = time.perf_counter()
                        for _ in range(calls): action(fn)
                        timings[name].append((time.perf_counter()-start)/calls)
                results.append(dict(shape=list(shape),dtype=str(dtype),mode=mode,samples_s=timings,
                    median_s={k:statistics.median(v) for k,v in timings.items()}))
    # A complete convolution -> normalization -> activation -> loss -> SGD step.
    shape = (2,4,12,12)
    for dtype in [torch.float32,torch.float64]:
        arrays = [rng.normal(size=(8,4,3,3)).astype(dtype)*.1,np.ones(8,dtype=dtype),np.zeros(8,dtype=dtype)]
        x = torch.tensor(rng.normal(size=shape).astype(dtype))
        target = torch.tensor(rng.normal(size=(2,8,12,12)).astype(dtype))
        def make(fn):
            params = [torch.nn.Parameter(torch.tensor(a)) for a in arrays]
            optimizer = torch.optim.SGD(params,lr=.01,momentum=.7)
            def step():
                optimizer.zero_grad()
                value = torch.nn.functional.conv2d(x,params[0],padding=1)
                loss = (torch.nn.functional.gelu(fn(value,*params[1:]))-target).square().mean()
                loss.backward();optimizer.step()
                return loss,params
            return step
        left,right = make(composed),make(direct)
        for _ in range(3):
            a,ap = left();b,bp = right()
            for v,w in [(a,b),*zip(ap,bp)]: np.testing.assert_allclose(v.numpy(),w.numpy(),rtol=3e-5 if dtype==torch.float32 else 3e-11,atol=3e-6 if dtype==torch.float32 else 3e-11)
        timings = {'composed':[],'direct':[]}
        for sample in range(samples):
            steps = [('composed',make(composed)),('direct',make(direct))]
            if (sample+int(reverse_order))%2: steps.reverse()
            for name,step in steps:
                step();start = time.perf_counter()
                for _ in range(max(3,calls//4)): step()
                timings[name].append((time.perf_counter()-start)/max(3,calls//4))
        results.append(dict(shape=list(shape),dtype=str(dtype),mode='cnn_sgd',samples_s=timings,
            median_s={k:statistics.median(v) for k,v in timings.items()}))
    return dict(status='success',baseline='existing BatchNorm composition; no old InstanceNorm API',
        correctness='values, JVP, reverse-over-forward, three SGD updates',samples=samples,calls=calls,flows=results)


if __name__ == '__main__': print(json.dumps(run(),indent=2))
