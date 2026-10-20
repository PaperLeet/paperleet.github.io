"""Compare supported 3D APIs, including overlap-heavy complete training flows.

Keep beside bench_max_pooling_1d.py and bench_max_pooling_3d.py. Both source
directories must support MaxPool3d (baseline >=0.43). No absent API is timed.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np

from bench_max_pooling_1d import arrays,load,measure
from bench_max_pooling_3d import make_flow as ordinary_flow,make_training as ordinary_training,pool


def make_flow(api,kind,mode,dtype):
    if kind != 'large_overlap':return ordinary_flow(api,kind,mode,True,dtype)
    rng=np.random.default_rng(4401);a=rng.normal(size=(2,8,12,24,32)).astype(dtype)
    if mode=='strided':a=np.repeat(a,2,axis=-1)[...,::2]
    if mode=='channels_last':a=a.transpose(0,2,3,4,1).copy().transpose(0,4,1,2,3)
    x=api.from_numpy(a);function=pool(api,'overlap',True,mode=='indices')
    if mode=='backward':
        x.requires_grad_()
        def run():
            x.grad=None;y=function(x);y.square().sum().backward()
            return y,x.grad
    elif mode=='jvp':
        v=api.from_numpy(rng.normal(size=a.shape).astype(dtype))
        def run():return api.func.jvp(function,(x,),(v,))
    elif mode=='indices':
        def run():return function(x)
    else:
        def run():return (function(x),)
    return run


def make_training(api,kind,dtype):
    if kind=='cnn':return ordinary_training(api,True,dtype)
    rng=np.random.default_rng(4402)
    model=api.nn.Sequential(api.nn.Conv3d(2,3,3,padding=1),api.nn.SiLU(),api.nn.MaxPool3d(3,1,1),
                             api.nn.Flatten(),api.nn.Linear(192,2)).to(dtype=getattr(api,np.dtype(dtype).name))
    model.load_state_dict({name:api.from_numpy(rng.normal(scale=.1,size=value.shape).astype(dtype))
                           for name,value in model.state_dict().items()})
    x=api.from_numpy(rng.normal(size=(2,2,4,4,4)).astype(dtype));target=api.from_numpy(rng.normal(size=(2,2)).astype(dtype))
    optimizer=api.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    def run():
        optimizer.zero_grad();loss=(model(x)-target).square().mean();loss.backward();optimizer.step()
        return loss,*tuple(model.parameters())
    return run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source',required=True);parser.add_argument('--candidate-source',required=True)
    parser.add_argument('--samples',type=int,default=7);parser.add_argument('--iterations',type=int,default=40)
    parser.add_argument('--reverse',action='store_true');parser.add_argument('--output',required=True)
    args=parser.parse_args();assert args.samples>0 and args.iterations>0
    baseline,bh=load('separable_baseline',args.baseline_source);candidate,ch=load('separable_candidate',args.candidate_source)
    control,control_hashes=load('separable_control',args.candidate_source);assert control_hashes==ch
    importlib.import_module(control.__name__+'._pooling3d')._SEPARABLE_MODE='off'
    variants=[('baseline_direct',baseline.torch),('candidate_control',control.torch),('candidate_direct',candidate.torch)]
    if args.reverse:variants.reverse()
    results=[]
    for dtype in (np.float32,np.float64):
        cases=[(kind,mode) for kind in ('cube','slab','overlap','dilated','large_cube','large_overlap')
               for mode in ('forward','strided','channels_last','indices','backward','jvp')]+[('cnn','training'),('cnn_overlap','training')]
        for kind,mode in cases:
            functions={name:make_training(api,kind,dtype) if mode=='training' else make_flow(api,kind,mode,dtype)
                       for name,api in variants}
            checked={name:arrays(fn()) for name,fn in functions.items()};expected=checked['baseline_direct']
            for values in checked.values():
                assert len(values)==len(expected)
                for a,b in zip(values,expected):np.testing.assert_allclose(a,b,rtol=4e-5,atol=4e-6)
            iterations=max(2,args.iterations//8) if mode=='training' else max(2,args.iterations//10) if kind=='large_overlap' else args.iterations
            timings={name:measure(fn,iterations,args.samples) for name,fn in functions.items()}
            results.append(dict(dtype=np.dtype(dtype).name,kind=kind,mode=mode,verified_arrays=len(expected),timings=timings))
    report=dict(status='success',python=sys.version,numpy=np.__version__,baseline_version=baseline.__version__,candidate_version=candidate.__version__,
                source_hashes=dict(baseline=bh,candidate=ch),order=[name for name,_ in variants],results=results,
                scope='76 complete flows using supported3D APIs on both releases. Candidate control disables separable dispatch only. Includes small/larger overlap,indices,VJP,JVP,strided/NDHWC inputs and nonoverlap/overlap Conv3d-SiLU-pool-linear-loss-momentum-SGD. Every result and one training step verified before timing. No native Torch/GPU/transfer or uniform-speedup claim.')
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='success',flows=len(results),output=args.output)))


if __name__=='__main__':main()
