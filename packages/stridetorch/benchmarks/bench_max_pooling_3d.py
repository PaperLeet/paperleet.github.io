"""Measure complete 3D flows against supported separable 2D/1D pooling.

Run after correctness gates, with --baseline-source and --candidate-source
pointing at standalone source directories. This does not time a missing old API.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from bench_max_pooling_1d import arrays, load, measure


def pool(api, kind, direct, indices=False, channels_last=False):
    k,s,p,d = {'cube':((2,2,2),(2,2,2),(0,0,0),(1,1,1)),
               'large_cube':((2,2,2),(2,2,2),(0,0,0),(1,1,1)),
               'slab':((1,2,2),(1,2,2),(0,0,0),(1,1,1)),
               'overlap':((3,3,3),(1,1,1),(1,1,1),(1,1,1)),
               'dilated':((2,2,2),(1,2,2),(1,1,1),(2,2,2))}[kind]
    if direct: return lambda x:api.nn.functional.max_pool3d(x,k,s,p,d,return_indices=indices)
    def restore(x):
        if channels_last:
            return x.permute(0,2,3,4,1).contiguous().permute(0,4,1,2,3)
        return x.contiguous()
    def function(x):
        prefix = x.shape[:-3];depth,height,width = x.shape[-3:];axis = len(prefix)
        result = api.nn.functional.max_pool2d(x.reshape(-1,1,height,width),k[1:],s[1:],p[1:],d[1:],return_indices=indices)
        y,hw_index = result if indices else (result,None)
        oh,ow = y.shape[-2:];y = y.reshape(*prefix,depth,oh,ow)
        y = y.permute(*range(axis),axis+1,axis+2,axis).reshape(-1,1,depth)
        result = api.nn.functional.max_pool1d(y,k[0],s[0],p[0],d[0],return_indices=indices)
        y,depth_index = result if indices else (result,None);od = y.shape[-1]
        def reorder(z):
            return z.reshape(*prefix,oh,ow,od).permute(*range(axis),axis+2,axis,axis+1)
        y = restore(reorder(y))
        if not indices: return y
        depth_index = reorder(depth_index)
        combined = api.gather(hw_index.reshape(*prefix,depth,oh,ow),axis,depth_index) + depth_index*(height*width)
        return y,restore(combined)
    return function


def make_flow(api, kind, mode, direct, dtype):
    rng = np.random.default_rng(4304)
    shape = (2,8,16,24,32) if kind == 'large_cube' else (2,3,6,8,10)
    a = rng.normal(size=shape).astype(dtype)
    if mode == 'strided': a = np.repeat(a,2,axis=-1)[...,::2]
    if mode == 'channels_last': a = a.transpose(0,2,3,4,1).copy().transpose(0,4,1,2,3)
    x = api.from_numpy(a);function = pool(api,kind,direct,mode == 'indices',mode == 'channels_last')
    if mode == 'backward':
        x.requires_grad_()
        def run():
            x.grad = None;y = function(x);y.square().sum().backward()
            return y,x.grad
    elif mode == 'indices':
        def run(): return function(x)
    elif mode == 'jvp':
        v = api.from_numpy(rng.normal(size=shape).astype(dtype))
        def run(): return api.func.jvp(function,(x,),(v,))
    else:
        def run(): return (function(x),)
    return run


def make_training(api, direct, dtype):
    rng = np.random.default_rng(4305)
    composition = pool(api,'cube',False)
    class Fixed(api.nn.Module):
        def forward(self,x): return composition(x)
    model = api.nn.Sequential(api.nn.Conv3d(2,3,3,padding=1),api.nn.SiLU(),
                              api.nn.MaxPool3d(2) if direct else Fixed(),
                              api.nn.Flatten(),api.nn.Linear(24,2)).to(dtype=getattr(api,np.dtype(dtype).name))
    model.load_state_dict({name:api.from_numpy(rng.normal(scale=.1,size=value.shape).astype(dtype))
                           for name,value in model.state_dict().items()})
    x = api.from_numpy(rng.normal(size=(2,2,4,4,4)).astype(dtype))
    target = api.from_numpy(rng.normal(size=(2,2)).astype(dtype))
    opt = api.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    def run():
        opt.zero_grad();loss = (model(x)-target).square().mean();loss.backward();opt.step()
        return loss,*tuple(model.parameters())
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source',required=True);parser.add_argument('--candidate-source',required=True)
    parser.add_argument('--samples',type=int,default=7);parser.add_argument('--iterations',type=int,default=40)
    parser.add_argument('--reverse',action='store_true');parser.add_argument('--output',required=True)
    args = parser.parse_args();assert args.samples > 0 and args.iterations > 0
    baseline,old_hashes = load('pooling_baseline',args.baseline_source)
    candidate,new_hashes = load('pooling_candidate',args.candidate_source)
    variants = [('baseline_composition',baseline.torch,False),('candidate_composition',candidate.torch,False),
                ('candidate_direct',candidate.torch,True)]
    if args.reverse: variants.reverse()
    results = []
    for dtype in (np.float32,np.float64):
        cases = [(kind,mode) for kind in ('cube','slab','overlap','dilated','large_cube')
                 for mode in ('forward','strided','channels_last','indices','backward','jvp')] + [('cnn','training')]
        for kind,mode in cases:
            functions = {name:make_training(api,direct,dtype) if mode == 'training'
                         else make_flow(api,kind,mode,direct,dtype) for name,api,direct in variants}
            checked = {name:arrays(fn()) for name,fn in functions.items()};expected = checked['baseline_composition']
            for values in checked.values():
                assert len(values) == len(expected)
                for a,b in zip(values,expected): np.testing.assert_allclose(a,b,rtol=3e-5,atol=3e-6)
            iterations = max(2,args.iterations//8) if mode == 'training' else args.iterations
            timings = {name:measure(fn,iterations,args.samples) for name,fn in functions.items()}
            results.append(dict(dtype=np.dtype(dtype).name,kind=kind,mode=mode,verified_arrays=len(expected),timings=timings))
    report = dict(status='success',python=sys.version,numpy=np.__version__,baseline_version=baseline.__version__,
                  candidate_version=candidate.__version__,source_hashes=dict(baseline=old_hashes,candidate=new_hashes),
                  order=[name for name,_,_ in variants],results=results,
                  scope='62 complete flows versus supported0.42 separable2D/1D composition, plus candidate composition control. Includes index recombination,contiguous outputs,strided/NDHWC inputs,VJP,JVP and Conv3d-SiLU-pool-linear-loss-SGD. No native Torch/GPU/transfer speed claim.')
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='success',flows=len(results),output=args.output)))


if __name__ == '__main__': main()
