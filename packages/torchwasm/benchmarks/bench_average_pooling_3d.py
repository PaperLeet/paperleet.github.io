"""Full-flow 3D averages vs the existing 2D+1D composition, on CPU/Pyodide.

The composition is a supported floating-point baseline, not a bitwise substitute
for native 3D accumulation on exceptional or ill-conditioned inputs. Every flow
checks values/gradients (and a complete optimizer step) before collecting times.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np


def load(name,directory):
    directory=Path(directory)
    spec=importlib.util.spec_from_file_location(name,directory/'__init__.py',submodule_search_locations=[str(directory)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.suffix=='.py' or p.name in ('LICENSE','NOTICE')}
    return module,hashes


def pool(api,kind,direct):
    fixed=kind in ('fixed','overlap');kernel=2 if kind=='fixed' else 3
    stride=2 if kind=='fixed' else 1;padding=0 if kind=='fixed' else 1
    output={'irregular':(3,3,3),'upsample':(9,10,11),'global':(1,1,1)}.get(kind)
    if direct:
        return (lambda x:api.nn.functional.avg_pool3d(x,kernel,stride,padding)) if fixed else (lambda x:api.nn.functional.adaptive_avg_pool3d(x,output))
    def compose(x):
        n,c,d,h,w=x.shape
        planes=x.reshape(n*c*d,1,h,w)
        hw=api.nn.functional.avg_pool2d(planes,kernel,stride,padding) if fixed else api.nn.functional.adaptive_avg_pool2d(planes,output[1:])
        oh,ow=hw.shape[-2:]
        columns=hw.reshape(n,c,d,oh,ow).permute(0,1,3,4,2).reshape(n*c*oh*ow,1,d)
        depth=api.nn.functional.avg_pool1d(columns,kernel,stride,padding) if fixed else api.nn.functional.adaptive_avg_pool1d(columns,output[0])
        return depth.reshape(n,c,oh,ow,depth.shape[-1]).permute(0,1,4,2,3).contiguous()
    return compose


def make_flow(api,kind,mode,direct,dtype):
    rng=np.random.default_rng(4504);a=rng.normal(size=(2,3,8,8,8)).astype(dtype)
    if mode=='strided':a=np.repeat(a,2,axis=-1)[...,::2]
    x=api.from_numpy(a);function=pool(api,kind,direct)
    if mode=='backward':
        x.requires_grad_()
        def run():
            x.grad=None;y=function(x);y.square().mean().backward();return y,x.grad
    elif mode=='jvp':
        direction=api.from_numpy(rng.normal(size=a.shape).astype(dtype))
        def run():return api.func.jvp(function,(x,),(direction,))
    else:
        def run():return (function(x),)
    return run


def make_training(api,direct,dtype):
    rng=np.random.default_rng(4505)
    class Pool(api.nn.Module):
        def __init__(self,kind):super().__init__();self.function=pool(api,kind,direct)
        def forward(self,x):return self.function(x)
    model=api.nn.Sequential(api.nn.Conv3d(2,3,3,padding=1),api.nn.SiLU(),Pool('fixed'),Pool('global'),api.nn.Flatten(),api.nn.Linear(3,2))
    model.to(dtype=getattr(api,np.dtype(dtype).name))
    model.load_state_dict({name:api.from_numpy(rng.normal(scale=.1,size=value.shape).astype(dtype)) for name,value in model.state_dict().items()})
    x=api.from_numpy(rng.normal(size=(2,2,8,8,8)).astype(dtype));target=api.from_numpy(rng.normal(size=(2,2)).astype(dtype))
    optimizer=api.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    def run():
        optimizer.zero_grad();y=model(x);loss=(y-target).square().mean();loss.backward();optimizer.step()
        return loss,*tuple(model.parameters())
    return run


def measure(function,iterations,samples):
    for _ in range(3):function()
    values=[]
    for _ in range(samples):
        start=time.perf_counter()
        for _ in range(iterations):function()
        values.append((time.perf_counter()-start)*1e6/iterations)
    return dict(median_us=statistics.median(values),samples_us=values,iterations=iterations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source',required=True);parser.add_argument('--candidate-source',required=True)
    parser.add_argument('--iterations',type=int,default=12);parser.add_argument('--samples',type=int,default=5)
    parser.add_argument('--output',required=True);parser.add_argument('--reverse',action='store_true')
    args=parser.parse_args();assert args.iterations>0 and args.samples>0
    baseline,old_hashes=load('average_baseline',args.baseline_source);candidate,new_hashes=load('average_candidate',args.candidate_source)
    variants=[('baseline_composition',baseline.torch,False),('candidate_composition',candidate.torch,False),('candidate_direct',candidate.torch,True)]
    if args.reverse:variants.reverse()
    results=[]
    for dtype in (np.float32,np.float64):
        cases=[(kind,mode) for kind in ('fixed','overlap','irregular','upsample','global') for mode in ('forward','strided','backward','jvp')]+[('cnn','training')]
        for kind,mode in cases:
            functions={name:make_training(api,direct,dtype) if mode=='training' else make_flow(api,kind,mode,direct,dtype) for name,api,direct in variants}
            checked={name:[v.detach().numpy().copy() for v in function()] for name,function in functions.items()}
            for name,values in checked.items():
                assert len(values)==len(checked['baseline_composition'])
                for actual,expected in zip(values,checked['baseline_composition']):np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-6)
            iterations=max(3,args.iterations//4) if mode=='training' else args.iterations
            timings={name:measure(function,iterations,args.samples) for name,function in functions.items()}
            results.append(dict(dtype=np.dtype(dtype).name,kind=kind,mode=mode,timings=timings))
    Path(args.output).write_text(json.dumps(dict(status='success',numpy=np.__version__,baseline_version=baseline.__version__,candidate_version=candidate.__version__,source_hashes=dict(baseline=old_hashes,candidate=new_hashes),results=results,scope='Finite random floating inputs; complete call/backward/JVP and CNN SGD step. Existing 2D+1D baseline and same-source composition control. No native Torch or GPU timing, no bitwise equivalence claim.'),indent=2)+'\n')
    print('PASS',len(results),'verified volumetric average flows')


if __name__=='__main__':main()
