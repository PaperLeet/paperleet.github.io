"""Compare direct3D pooling with native-order inverse adjoints on CPU/Pyodide."""
import argparse,json
from pathlib import Path
import numpy as np
from bench_average_pooling_3d import load,make_flow,make_training,measure


def training(api,output,dtype):
    rng=np.random.default_rng(4601)
    model=api.nn.Sequential(api.nn.Conv3d(2,3,3,padding=1),api.nn.SiLU(),api.nn.AvgPool3d(2),api.nn.AdaptiveAvgPool3d(output),api.nn.Flatten(),api.nn.Linear(3*int(np.prod(output)),2))
    model.to(dtype=getattr(api,np.dtype(dtype).name))
    model.load_state_dict({name:api.from_numpy(rng.normal(scale=.1,size=v.shape).astype(dtype)) for name,v in model.state_dict().items()})
    x=api.from_numpy(rng.normal(size=(2,2,8,8,8)).astype(dtype));target=api.from_numpy(rng.normal(size=(2,2)).astype(dtype));opt=api.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    def run():
        opt.zero_grad();loss=(model(x)-target).square().mean();loss.backward();opt.step();return loss,*tuple(model.parameters())
    return run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-source',required=True);parser.add_argument('--candidate-source',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--samples',type=int,default=5);parser.add_argument('--iterations',type=int,default=12);parser.add_argument('--reverse',action='store_true');args=parser.parse_args()
    baseline,old_hashes=load('inverse_baseline',args.baseline_source);candidate,new_hashes=load('inverse_candidate',args.candidate_source);control,control_hashes=load('inverse_control',args.candidate_source)
    assert control_hashes==new_hashes
    for module in (baseline,candidate,control):module.torch
    control._average_pooling3d._INVERSE_MODE='off'
    variants=[('baseline_direct',baseline),('candidate_inverse',candidate),('candidate_control',control)]
    if args.reverse:variants.reverse()
    results=[]
    for dtype in (np.float32,np.float64):
        cases=[(kind,mode) for kind in ('fixed','overlap','irregular','upsample','global') for mode in ('forward','strided','backward','jvp')]+[('cnn','training'),('cnn_irregular','training'),('cnn_upsample','training')]
        for kind,mode in cases:
            functions={name:(training(module.torch,(3,3,3) if kind=='cnn_irregular' else (5,6,7),dtype) if kind.startswith('cnn_') else make_training(module.torch,True,dtype)) if mode=='training' else make_flow(module.torch,kind,mode,True,dtype) for name,module in variants}
            checked={};routes={}
            for name,module in variants:
                counts=dict(selected=0,fallback=0)
                original=getattr(module._average_pooling3d,'_inverse_backward',None)
                if original is not None:
                    def wrapped(*args,_original=original,_counts=counts):
                        value=_original(*args);_counts['fallback' if value is None else 'selected']+=1;return value
                    module._average_pooling3d._inverse_backward=wrapped
                try:checked[name]=[v.detach().numpy().copy() for v in functions[name]()]
                finally:
                    if original is not None:module._average_pooling3d._inverse_backward=original
                routes[name]=counts
            for name,values in checked.items():
                assert len(values)==len(checked['baseline_direct'])
                for actual,expected in zip(values,checked['baseline_direct']):np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-6)
            iterations=max(3,args.iterations//4) if mode=='training' else args.iterations
            timings={name:measure(function,iterations,args.samples) for name,function in functions.items()}
            results.append(dict(dtype=np.dtype(dtype).name,kind=kind,mode=mode,timings=timings,qualification_inverse_calls=routes))
    Path(args.output).write_text(json.dumps(dict(status='success',scope='Every finite-random direct3D value/gradient/SGD step checked against baseline before timing; same-source inverse-disabled control. Actual selected/fallback route counts are qualified before timing. No native Torch/GPU timing or cross-runtime speed claim.',numpy=np.__version__,baseline_version=baseline.__version__,candidate_version=candidate.__version__,source_hashes=dict(baseline=old_hashes,candidate=new_hashes),results=results),indent=2)+'\n')
    print('PASS',len(results),'verified direct3D flows')


if __name__=='__main__':main()
