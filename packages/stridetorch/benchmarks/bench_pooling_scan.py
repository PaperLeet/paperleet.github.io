"""Compare forced ordered max-pool scans on identical padded tiny inputs.

This targets the scan fallback, not automatic whole-application throughput.
Both sources use the same private algorithm override; ordinary dispatch is
measured separately by bench_max_pooling_1d.py. Million-tap kernels are covered
by correctness tests; timed baselines use bounded 1,024/4,096-tap kernels.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source',required=True)
    parser.add_argument('--candidate-source',required=True)
    parser.add_argument('--iterations',type=int,default=4)
    parser.add_argument('--samples',type=int,default=7)
    parser.add_argument('--reverse',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();assert args.iterations>0 and args.samples>0
    helper=Path(__file__).with_name('bench_max_pooling_1d.py')
    spec=importlib.util.spec_from_file_location('pooling_benchmark_helpers',helper)
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    baseline,old_hashes=common.load('scan_baseline',args.baseline_source)
    candidate,new_hashes=common.load('scan_candidate',args.candidate_source)
    for package in (baseline,candidate):
        importlib.import_module(package.__name__+'._pooling')._ALGORITHM='scan'
    variants=[('baseline_composition',baseline.torch,False),('candidate_composition',candidate.torch,False),('candidate_direct',candidate.torch,True)]
    if args.reverse:variants.reverse()
    results=[]
    for dtype in (np.float32,np.float64):
        for kernel in (1024,4096):
            for mode in ('forward','backward','jvp'):
                calls={}
                for name,api,direct in variants:
                    x=api.from_numpy(np.arange(7,dtype=dtype).reshape(1,1,7)).requires_grad_(mode=='backward')
                    if direct:function=lambda z,api=api,k=kernel:api.nn.functional.max_pool1d(z,k,1,k//2)
                    else:function=lambda z,api=api,k=kernel:api.nn.functional.max_pool2d(z.unsqueeze(-2),(1,k),(1,1),(0,k//2)).squeeze(-2)
                    if mode=='backward':
                        def call(x=x,function=function):
                            x.grad=None;y=function(x);y.sum().backward();return y,x.grad
                    elif mode=='jvp':
                        direction=api.ones_like(x)
                        def call(x=x,function=function,direction=direction,api=api):return api.func.jvp(function,(x,),(direction,))
                    else:
                        def call(x=x,function=function):return (function(x),)
                    calls[name]=call
                checked={name:common.arrays(call()) for name,call in calls.items()}
                for values in checked.values():
                    for aa,bb in zip(values,checked['baseline_composition']):np.testing.assert_array_equal(aa,bb)
                for values in checked.values():
                    np.testing.assert_array_equal(values[0],6.)
                    if mode=='backward':np.testing.assert_array_equal(values[1],[[[0,0,0,0,0,0,8]]])
                    if mode=='jvp':np.testing.assert_array_equal(values[1],1.)
                timings={name:common.measure(call,args.iterations,args.samples) for name,call in calls.items()}
                results.append(dict(dtype=np.dtype(dtype).name,kernel=kernel,mode=mode,timings=timings))
    report=dict(status='success',python=sys.version,numpy=np.__version__,source_hashes=dict(baseline=old_hashes,candidate=new_hashes),
                baseline_version=baseline.__version__,candidate_version=candidate.__version__,order=[v[0] for v in variants],
                helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),results=results,
                scope='Forced scan fallback only, same algorithm override and valid values/VJP/JVP. Existing 2D composition baseline; candidate composition/direct1D. No default-dispatch, application, native Torch or GPU speed claim.')
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='success',flows=len(results),output=args.output)))


if __name__=='__main__':main()
