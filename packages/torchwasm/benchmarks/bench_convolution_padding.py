"""Compare two library snapshots and same-source zero-padding controls.

Only safe legacy halos are executed. Huge-span allocation safety is tested
separately; these timings do not measure RSS or imply GPU/native Torch speed.
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


def load(name,path):
    path=Path(path);spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.suffix=='.py' or p.name in ['LICENSE','NOTICE']}
    return package,hashes


def run(baseline_source,candidate_source,samples=5,calls=8,reverse_order=False):
    baseline,bh=load('_stride_pad_baseline',baseline_source);candidate,ch=load('_stride_pad_candidate',candidate_source)
    import importlib
    bc=importlib.import_module(baseline.__name__+'._convolution');cc=importlib.import_module(candidate.__name__+'._convolution')
    variants={'baseline':(baseline,bc,None),'candidate':(candidate,cc,'auto'),'control':(candidate,cc,'padded'),'gather':(candidate,cc,'gather'),'tap':(candidate,cc,'tap')}
    rng=np.random.default_rng(4913);results=[]
    cases=[('halo1d',(2,4,3),8,65536),('halo2d',(2,4,3,3),8,127),('halo3d',(2,4,3,3,3),8,31),('ordinary',(2,4,8,8),8,1),('transpose_control',(2,4,8,8),6,1)]
    for kind,shape,co,halo in cases:
        dim=len(shape)-2;transposed=kind=='transpose_control';names=list(variants) if not transposed else ['baseline','candidate','control']
        for dtype in ['float32','float64']:
            raw=[rng.normal(size=shape).astype(dtype),rng.normal(size=((shape[1],co) if transposed else (co,shape[1]))+((3,)*dim)).astype(dtype),rng.normal(size=co).astype(dtype)]
            directions=[rng.normal(size=a.shape).astype(dtype) for a in raw]
            def configure(name):
                package,module,mode=variants[name]
                if mode is not None:module._PADDING_ALGORITHM=mode
                return package.torch
            def function(t,x,w,b):
                return t.nn.functional.conv_transpose2d(x,w,b,stride=2,padding=1,output_padding=1) if transposed else getattr(t.nn.functional,f'conv{dim}d')(x,w,b,padding=halo,dilation=halo)
            seeds=None;qualification={}
            def check(name):
                nonlocal seeds
                t=configure(name);values=tuple(t.tensor(a,requires_grad=True) for a in raw);ds=tuple(t.tensor(a,requires_grad=True) for a in directions)
                module=importlib.import_module(variants[name][0].__name__+('._transpose_convolution' if transposed else '._convolution'))
                q=dict(matrix=0,direct=0,padding={});selector=module._use_matrix
                def select(*args,**kwargs):
                    result=selector(*args,**kwargs);q['matrix' if result else 'direct']+=1;return result
                strategy=getattr(module,'_padding_strategy',None)
                def select_padding(*args):
                    result=strategy(*args);q['padding'][result]=q['padding'].get(result,0)+1;return result
                module._use_matrix=select
                if strategy is not None:module._padding_strategy=select_padding
                try:
                    p,j=t.func.jvp(lambda x,w,b:function(t,x,w,b),values,ds)
                    if seeds is None:seeds=[rng.normal(size=p.shape).astype(dtype),rng.normal(size=p.shape).astype(dtype)]
                    (p*t.tensor(seeds[0])+j*t.tensor(seeds[1])).sum().backward()
                    qualification[name]=q
                    return [v.numpy().copy() for v in (p,j,*[x.grad for x in (*values,*ds)])]
                finally:
                    module._use_matrix=selector
                    if strategy is not None:module._padding_strategy=strategy
            expected=check('baseline')
            for name in names[1:]:
                for actual,reference in zip(check(name),expected):
                    np.testing.assert_allclose(actual,reference,rtol=2e-4 if dtype=='float32' else 2e-11,atol=5e-5 if dtype=='float32' else 2e-11)
            for mode in ['forward','reverse','jvp']:
                def action(name):
                    t=configure(name)
                    if mode=='forward':
                        with t.no_grad():return function(t,*[t.tensor(a) for a in raw])
                    values=tuple(t.tensor(a,requires_grad=True) for a in raw)
                    if mode=='reverse':y=function(t,*values);y.backward(t.tensor(seeds[0]));return y
                    return t.func.jvp(lambda x,w,b:function(t,x,w,b),values,tuple(t.tensor(a) for a in directions))
                for name in names:action(name)
                times={name:[] for name in names}
                for sample in range(samples):
                    order=names.copy()
                    if (sample+int(reverse_order))%2:order.reverse()
                    for name in order:
                        start=time.perf_counter()
                        for _ in range(calls):action(name)
                        times[name].append((time.perf_counter()-start)/calls)
                results.append(dict(kind=kind,shape=list(shape),dtype=dtype,mode=mode,qualification=qualification,samples_s=times,median_s={k:statistics.median(v) for k,v in times.items()}))
    for dtype in ['float32','float64']:
        arrays=[rng.normal(size=(4,2,3,3,3)).astype(dtype)*.1,np.ones(4,dtype=dtype),np.zeros(4,dtype=dtype),rng.normal(size=(4,2,1,1,1)).astype(dtype)*.1]
        raw=rng.normal(size=(1,2,3,3,3)).astype(dtype);target=rng.normal(size=raw.shape).astype(dtype)
        def make(name):
            t=configure(name);params=[t.nn.Parameter(t.tensor(a)) for a in arrays];opt=t.optim.SGD(params,lr=.01,momentum=.7);x=t.tensor(raw,requires_grad=True);goal=t.tensor(target)
            def step():
                configure(name);opt.zero_grad();x.grad=None
                value=t.nn.functional.conv3d(x,params[0],padding=31,dilation=31)
                value=t.nn.functional.gelu(t.nn.functional.instance_norm(value,weight=params[1],bias=params[2]))
                value=t.nn.functional.conv_transpose3d(value,params[3]);loss=(value-goal).square().mean();loss.backward();opt.step()
                return [loss.numpy().copy(),*[p.numpy().copy() for p in params],x.grad.numpy().copy()]
            return step
        expected=[];step=make('baseline')
        for _ in range(3):expected.append(step())
        for name in list(variants)[1:]:
            step=make(name)
            for reference in expected:
                for a,b in zip(step(),reference):np.testing.assert_allclose(a,b,rtol=5e-5 if dtype=='float32' else 2e-11,atol=5e-6 if dtype=='float32' else 2e-11)
        times={name:[] for name in variants}
        for sample in range(samples):
            order=list(variants)
            if (sample+int(reverse_order))%2:order.reverse()
            for name in order:
                step=make(name);step();start=time.perf_counter()
                for _ in range(max(3,calls//3)):step()
                times[name].append((time.perf_counter()-start)/max(3,calls//3))
        results.append(dict(kind='training',shape=list(raw.shape),dtype=dtype,mode='encoder_decoder_sgd',samples_s=times,median_s={k:statistics.median(v) for k,v in times.items()}))
    cc._PADDING_ALGORITHM='auto'
    return dict(status='success',baseline_version=baseline.__version__,candidate_version=candidate.__version__,source_hashes=dict(baseline=bh,candidate=ch),samples=samples,calls=calls,flows=results,scope='Direct version comparison plus same-source padded/gather/tap controls; finite values/VJP/JVP/mixed pullbacks and3SGD updates checked before timing. Original ordinary and transposed-convolution controls retained. Only safe legacy halo allocations executed; no RSS/GPU/native Torch claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-source',required=True);parser.add_argument('--candidate-source',required=True);parser.add_argument('--output',required=True);parser.add_argument('--samples',type=int,default=5);parser.add_argument('--calls',type=int,default=8);parser.add_argument('--reverse',action='store_true');args=parser.parse_args()
    Path(args.output).write_text(json.dumps(run(args.baseline_source,args.candidate_source,args.samples,args.calls,args.reverse),indent=2)+'\n')
