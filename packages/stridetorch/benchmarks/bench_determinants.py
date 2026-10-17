"""Paired determinant workflows, stable-log compositions and graph controls."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import warnings

import numpy as np


def load(name,path):
    path=Path(path);spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.suffix=='.py' or p.name in ['LICENSE','NOTICE']}
    return module,hashes


def _data(shape,dtype):
    a=np.sin(np.arange(np.prod(shape))*.19+.2).reshape(shape)
    value=a@a.swapaxes(-1,-2)/shape[-1]+np.eye(shape[-1])
    if np.dtype(dtype).kind=='c':value=value+np.eye(shape[-1])*.15j
    return value.astype(dtype)


def _value(t,x,operation,baseline):
    if operation=='det':return t.det(x)
    if operation=='logdet':return t.det(x).log() if baseline else t.logdet(x)
    if operation=='slogdet':return t.det(x).abs().log() if baseline else t.slogdet(x).logabsdet
    value=x
    for _ in range(16):value=value*.9+.1
    return value.square().sum()


def _method(module,data,operation,backward,baseline):
    t=module.torch;x=t.tensor(data,requires_grad=backward);shape=() if operation=='graph' else data.shape[:-2]
    seed=np.sin(np.arange(np.prod(shape))+.3).reshape(shape).astype(data.dtype)+.5;weight=t.tensor(seed) if backward else None
    def call():
        x.grad=None;value=_value(t,x,operation,baseline)
        if backward:(value*weight).sum().backward()
        return value
    def capture():
        value=call();return [value.numpy().copy(),x.grad.numpy().copy()] if backward else [value.numpy().copy()]
    return call,capture


def _compare(actual,want,dtype,matched):
    assert len(actual)==len(want)
    for i,(a,b) in enumerate(zip(actual,want)):
        assert a.shape==b.shape
        if matched:assert a.dtype==b.dtype
        if dtype=='float32' and i==1:
            # Old SVD cofactors and adjoint solves can differ in cancellation-
            # prone components. Native/independent accuracy is checked separately.
            delta=a.astype('float64')-b.astype('float64');tiny=np.finfo('float64').tiny
            assert np.linalg.norm(delta)/max(np.linalg.norm(b.astype('float64')),tiny)<=1e-5
            assert np.max(np.abs(delta),initial=0)/max(np.max(np.abs(b),initial=0),tiny)<=1e-5
        else:np.testing.assert_allclose(a,b,rtol=3e-5 if dtype in ['float32','complex64'] else 3e-11,atol=3e-6 if dtype in ['float32','complex64'] else 3e-11,equal_nan=True)


def run(baseline_source,candidate_source,samples=5):
    old,oh=load('_determinant_baseline',baseline_source);new,nh=load('_determinant_candidate',candidate_source)
    warnings.filterwarnings('ignore',category=RuntimeWarning);plans=[];rows=[]
    shapes=[('scalar',(1,1)),('small',(2,2)),('medium',(8,8)),('large',(32,32)),('larger',(64,64)),('batch',(4,8,8)),('batch_large',(4,32,32))]
    for dtype in ['float32','float64']:
        for label,shape in shapes:
            for operation in ['det','logdet','slogdet']:
                for backward in [False,True]:plans.append((operation,dtype,label,shape,backward,True))
        for label,shape in [('graph_scalar',(1,)),('graph_vector',(32,))]:plans.append(('graph',dtype,label,shape,True,True))
    for dtype in ['complex64','complex128']:
        for label,shape in [('small',(2,2)),('medium',(8,8)),('large',(32,32))]:
            for operation in ['logdet','slogdet']:plans.append((operation,dtype,label,shape,False,False))
    for operation,dtype,label,shape,backward,matched in plans:
        data=np.full(shape,.3,dtype=dtype) if operation=='graph' else _data(shape,dtype)
        variants={'candidate':_method(new,data,operation,backward,False)}
        if matched:variants['baseline']=_method(old,data,operation,backward,True)
        actual=variants['candidate'][1]()
        if matched:want=variants['baseline'][1]()
        else:
            sign,logabs=np.linalg.slogdet(data);value=logabs if operation=='slogdet' else logabs.astype(data.dtype)+np.asarray(1j,dtype=data.dtype)*np.angle(sign);want=[np.asarray(value)]
        _compare(actual,want,dtype,matched);del actual,want
        methods={key:method[0] for key,method in variants.items()}
        for fn in methods.values():fn()
        gc.collect();calibration={};calls=1;times={key:[] for key in methods}
        for key,fn in methods.items():
            batch=1
            while True:
                start=time.perf_counter()
                for _ in range(batch):fn()
                elapsed=time.perf_counter()-start
                if elapsed>=.02:break
                if batch>=1_048_576:raise RuntimeError('Determinant benchmark timer resolution is insufficient')
                batch*=2
            calls=max(calls,batch);calibration[key]=dict(calls=batch,elapsed_s=elapsed)
        for repeat in range(samples):
            order=list(methods) if repeat%2==0 else list(reversed(methods))
            for key in order:
                start=time.perf_counter()
                for _ in range(calls):methods[key]()
                times[key].append((time.perf_counter()-start)/calls)
        medians={key:statistics.median(value) for key,value in times.items()}
        rows.append(dict(operation=operation,dtype=dtype,shape=label,input_shape=list(shape),mode='full_step' if backward else 'forward',matched=matched,baseline_kind='det.log' if operation=='logdet' else 'det.abs.log' if operation=='slogdet' else operation,calls=calls,calibration=calibration,samples_s=times,median_s=medians,candidate_over_baseline=medians['candidate']/medians['baseline'] if matched else None))
    assert len(rows)==100 and sum(v['matched'] for v in rows)==88
    return dict(versions=dict(baseline=old.__version__,candidate=new.__version__),source_hashes=dict(baseline=oh,candidate=nh),samples=samples,rows=rows,scope='88 matched ordinary-SPD determinant/log-composition or graph-control rows against 0.60; 12 candidate-only complex-forward rows. Inputs, conversions and seeds are outside timing; dispatch, graph construction, kernels, optional weighted backward, gradient clearing and result destruction are inside. Five alternating samples in common calibrated batches >=20 ms. Float32 old/new gradients use a separately labeled 1e-5 normwise comparison; native accuracy is independently qualified. No native-Torch/GPU, cold-start or universal-throughput claim.')


def memory(baseline_source,candidate_source):
    import tracemalloc
    old,oh=load('_determinant_memory_baseline',baseline_source);new,nh=load('_determinant_memory_candidate',candidate_source);rows=[]
    for label,module in [('baseline',old),('candidate',new)]:
        t=module.torch
        for dtype in ['float32','float64']:
            for shape in [(64,64),(4,32,32)]:
                for operation in ['det','logdet','slogdet']:
                    x=t.tensor(_data(shape,dtype),requires_grad=True);weight=t.ones(shape[:-2],dtype=getattr(t,dtype))
                    gc.collect();tracemalloc.start();value=_value(t,x,operation,label=='baseline');retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                    tracemalloc.start();(value*weight).sum().backward();_,back_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                    rows.append(dict(variant=label,operation=operation,dtype=dtype,input_shape=list(shape),new_retained_bytes=retained,forward_peak_bytes=peak,backward_peak_bytes=back_peak,result_bytes=value.numel()*value.element_size()))
                    del x,weight,value;gc.collect()
    return dict(source_hashes=dict(baseline=oh,candidate=nh),rows=rows,scope='CPU tracemalloc incremental tracked allocations; inputs/seeds allocated before tracing. Forward retained/peak and separate weighted-backward peak. Not total tape memory, process RSS, LAPACK workspace or WASM heap size.')


if __name__=='__main__':
    result=memory(sys.argv[1],sys.argv[2]) if '--memory' in sys.argv[3:] else run(sys.argv[1],sys.argv[2])
    print(json.dumps(result,indent=2))
