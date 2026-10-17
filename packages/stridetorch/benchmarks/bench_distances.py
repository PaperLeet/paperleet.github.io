"""Matched distance paths, non-Euclidean orders and feature-heavy workloads."""
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


def _data(shape,dtype,offset):return np.sin(np.arange(np.prod(shape))*.17+offset).reshape(shape).astype(dtype)


def _reference(a,b,p,seed,kind):
    if kind=='pdist':
        i,j=np.triu_indices(len(a),k=1);delta=a[i].astype('float64')-a[j].astype('float64')
    else:delta=a.astype('float64')[..., :,None,:]-b.astype('float64')[...,None,:,:]
    absolute=np.abs(delta)
    value=(absolute!=0).sum(-1).astype('float64') if p==0 else absolute.max(-1) if p==np.inf else np.sum(absolute**p,axis=-1)**(1/p)
    if seed is None:return [value]
    with np.errstate(all='ignore'):
        if p==0:unit=np.zeros_like(delta)
        elif p==np.inf:unit=np.sign(delta)*(absolute==value[...,None])
        else:unit=np.divide(np.sign(delta)*absolute**(p-1),value[...,None]**(p-1),out=np.zeros_like(delta),where=(absolute!=0)&(value[...,None]!=0))
    unit*=seed[...,None]
    if kind=='pdist':
        gradient=np.zeros_like(a,dtype='float64');np.add.at(gradient,i,unit);np.add.at(gradient,j,-unit);return [value,gradient]
    return [value,unit.sum(axis=-2),-unit.sum(axis=-3)]


def _method(module,a,b,p,kind,backward):
    t=module.torch;x=t.tensor(a,requires_grad=backward);y=t.tensor(b,requires_grad=backward) if kind=='cdist' else None
    shape=(*np.broadcast_shapes(a.shape[:-2],b.shape[:-2]),a.shape[-2],b.shape[-2]) if kind=='cdist' else (len(a)*(len(a)-1)//2,)
    seed=np.sin(np.arange(np.prod(shape))+.3).reshape(shape).astype(a.dtype);weight=t.tensor(seed) if backward else None
    def call():
        x.grad=None
        if y is not None:y.grad=None
        value=t.cdist(x,y,p=p,compute_mode='donot_use_mm_for_euclid_dist') if kind=='cdist' else t.pdist(x,p=p)
        if backward:(value*weight).sum().backward()
        return value
    def capture():
        value=call();arrays=[value.numpy().copy()]
        if backward:arrays.append(x.grad.numpy().copy())
        if backward and y is not None:arrays.append(y.grad.numpy().copy())
        return arrays
    return call,capture,seed


def run(baseline_source,candidate_source,samples=5):
    old,oh=load('_distance_baseline',baseline_source);new,nh=load('_distance_candidate',candidate_source)
    warnings.filterwarnings('ignore',category=RuntimeWarning);plans=[];rows=[]
    shapes=[('small',(4,7),(5,7)),('medium',(64,64),(80,64)),('large',(128,128),(192,128)),('broadcast',(2,1,16,32),(1,3,20,32))]
    for dtype in ['float32','float64']:
        for p in [1.,2.]:
            for label,left,right in shapes:
                for backward in [False,True]:plans.append(('cdist',dtype,p,label,left,right,backward,True))
        for label,shape in [('small',(5,7)),('medium',(64,64)),('large',(128,128))]:
            for backward in [False,True]:plans.append(('pdist',dtype,2.,label,shape,shape,backward,True))
    for p in [0.,.5,1.5,3.,np.inf]:
        for backward in [False,True]:plans.append(('cdist','float32',p,'non_euclidean_order',(16,32),(20,32),backward,True))
    for p in [1.,3.]:
        for backward in [False,True]:plans.append(('pdist','float32',p,'non_euclidean_order',(64,64),(64,64),backward,True))
    for dtype in ['float32','float64']:
        for backward in [False,True]:plans.append(('cdist',dtype,2.,'larger_feature_workload_ones',(128,1024),(129,1024),backward,True))
    for kind,dtype,p,label,left,right,backward,matched in plans:
        a=_data(left,dtype,.2);b=_data(right,dtype,.7)
        if label=='larger_feature_workload_ones':a.fill(1);b.fill(0)
        variants={'candidate':_method(new,a,b,p,kind,backward)}
        if matched:variants['baseline']=_method(old,a,b,p,kind,backward)
        actual=variants['candidate'][1]()
        if matched:want=variants['baseline'][1]()
        elif label=='larger_feature_workload_ones':
            want=[np.full((128,129),32.,dtype=dtype)]
            if backward:
                seed=variants['candidate'][2].astype('float64');want.extend([np.broadcast_to(seed.sum(-1)[:,None]/32,left),np.broadcast_to(-seed.sum(-2)[:,None]/32,right)])
        else:want=_reference(a,b,p,variants['candidate'][2].astype('float64') if backward else None,kind)
        assert len(actual)==len(want)
        tol=(3e-5,3e-5) if dtype=='float32' else (3e-11,3e-11)
        for x,y in zip(actual,want):
            assert x.shape==y.shape
            if matched:assert x.dtype==y.dtype
            np.testing.assert_allclose(x,y,rtol=tol[0],atol=tol[1])
        del actual,want
        methods={key:value[0] for key,value in variants.items()}
        for fn in methods.values():fn()
        gc.collect();calibration={};calls=1;times={key:[] for key in methods}
        for key,fn in methods.items():
            batch=1
            while True:
                start=time.perf_counter()
                for _ in range(batch):fn()
                elapsed=time.perf_counter()-start
                if elapsed>=.02:break
                if batch>=1_048_576:raise RuntimeError('Distance benchmark timer resolution is insufficient')
                batch*=2
            calls=max(calls,batch);calibration[key]=dict(calls=batch,elapsed_s=elapsed)
        for repeat in range(samples):
            order=list(methods) if repeat%2==0 else list(reversed(methods))
            for key in order:
                start=time.perf_counter()
                for _ in range(calls):methods[key]()
                times[key].append((time.perf_counter()-start)/calls)
        medians={key:statistics.median(value) for key,value in times.items()}
        rows.append(dict(factory=kind,dtype=dtype,p=str(p),shape=label,left_shape=list(left),right_shape=list(right),mode='backward' if backward else 'forward',matched=matched,calls=calls,calibration=calibration,samples_s=times,median_s=medians,candidate_over_baseline=medians['candidate']/medians['baseline'] if matched else None))
    assert len(rows)==62
    return dict(versions=dict(baseline=old.__version__,candidate=new.__version__),source_hashes=dict(baseline=oh,candidate=nh),samples=samples,rows=rows,
                scope='62 matched rows against 0.58, including non-Euclidean orders and larger-feature workloads. Inputs, seeds and conversion are outside timing; graph construction, distance kernels, optional weighted backward, gradient clearing and result destruction are included. Shared calibrated batches >=20 ms, five samples, alternating variant order. No native Torch/GPU, cold-load or universal throughput claim.')


def memory(baseline_source,candidate_source):
    import tracemalloc
    old,oh=load('_distance_memory_baseline',baseline_source);new,nh=load('_distance_memory_candidate',candidate_source);rows=[]
    for label,module in [('baseline',old),('candidate',new)]:
        t=module.torch
        for kind,shape in [('cdist',(128,128)),('pdist',(64,64)),('pdist',(128,128))]:
            x=t.tensor(_data(shape,'float32',.2),requires_grad=True);y=t.tensor(_data((192,128),'float32',.7),requires_grad=True)
            gc.collect();tracemalloc.start()
            value=t.cdist(x,y,compute_mode='donot_use_mm_for_euclid_dist') if kind=='cdist' else t.pdist(x)
            retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop();tracemalloc.start();value.sum().backward();_,backward_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            rows.append(dict(variant=label,factory=kind,input_shape=list(shape),new_retained_bytes=retained,forward_peak_bytes=peak,backward_peak_bytes=backward_peak,result_bytes=value.numel()*value.element_size()))
            del x,y,value;gc.collect()
    return dict(source_hashes=dict(baseline=oh,candidate=nh),rows=rows,scope='CPU tracemalloc incremental tracked allocations with inputs created before tracing; forward retained/peak and separate backward peak. Pre-existing input storage is excluded in both variants. Not total tape memory, process RSS or WASM heap size.')


if __name__=='__main__':
    result=memory(sys.argv[1],sys.argv[2]) if '--memory' in sys.argv[3:] else run(sys.argv[1],sys.argv[2])
    print(json.dumps(result,indent=2))
