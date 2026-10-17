"""Matched valid sequence workloads; new logspace latency is candidate-only."""
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
    path=Path(path)
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.suffix=='.py' or p.name in ['LICENSE','NOTICE']}
    return module,hashes


def run(baseline_source,candidate_source,samples=5):
    old,oh=load('_sequence_baseline',baseline_source);new,nh=load('_sequence_candidate',candidate_source)
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    rows=[]
    for name in ['arange','linspace','logspace']:
        for dtype in ['float16','float32','float64','int64']:
            if name=='logspace' and dtype not in ['float32','float64']:continue
            for size in [16,4096,1_000_000]:
                if name=='arange' and dtype=='float16' and size>65504:continue
                def factory(m):
                    t=m.torch;target=getattr(t,dtype)
                    if name=='arange':return lambda:t.arange(size,dtype=target)
                    end=size-1 if dtype=='int64' else 1.
                    if name=='linspace':return lambda:t.linspace(0.,end,size,dtype=target)
                    return lambda:t.logspace(-2.,2.,size,dtype=target)
                methods={'candidate':factory(new)}
                if name!='logspace':methods['baseline']=factory(old)
                actual=methods['candidate']().numpy()
                if 'baseline' in methods:want=methods['baseline']().numpy()
                else:want=np.logspace(-2.,2.,size,dtype=dtype)
                assert actual.shape==want.shape and actual.dtype==want.dtype
                tolerance=(2e-3,1e-6) if dtype=='float16' else (3e-7,1e-8) if dtype=='float32' else (4e-15,2e-16)
                if dtype=='int64':np.testing.assert_array_equal(actual,want)
                else:np.testing.assert_allclose(actual,want,rtol=tolerance[0],atol=tolerance[1])
                del actual,want
                for fn in methods.values():fn()
                gc.collect();times={key:[] for key in methods}
                # Browser clocks may be coarsened to 100 us. Calibrate outside
                # timing, then use one shared batch size with >=20 ms batches.
                calls=1;calibration={}
                for key,method in methods.items():
                    batch=1
                    while True:
                        start=time.perf_counter()
                        for _ in range(batch):method()
                        elapsed=time.perf_counter()-start
                        if elapsed>=.02:break
                        if batch>=1_048_576:raise RuntimeError('Timer resolution is insufficient for sequence benchmarking')
                        batch*=2
                    calls=max(calls,batch);calibration[key]=dict(calls=batch,elapsed_s=elapsed)
                for repeat in range(samples):
                    order=list(methods) if repeat%2==0 else list(reversed(methods))
                    for key in order:
                        start=time.perf_counter()
                        for _ in range(calls):methods[key]()
                        times[key].append((time.perf_counter()-start)/calls)
                medians={key:statistics.median(value) for key,value in times.items()}
                rows.append(dict(factory=name,dtype=dtype,size=size,calls=calls,calibration=calibration,samples_s=times,median_s=medians,
                                 candidate_over_baseline=medians['candidate']/medians['baseline'] if 'baseline' in medians else None))
    return dict(versions=dict(baseline=old.__version__,candidate=new.__version__),source_hashes=dict(baseline=oh,candidate=nh),samples=samples,rows=rows,
                scope='23 matched valid arange/linspace cases; 6 candidate-only logspace cases. Known incorrect integer-floor/rounded-step baseline inputs are excluded from speedup claims. Library construction and destruction included; correctness/warmup and GC outside timing. No native Torch/GPU, cold-load or universal throughput claim.')


def memory(baseline_source,candidate_source):
    import tracemalloc
    old,oh=load('_sequence_memory_baseline',baseline_source);new,nh=load('_sequence_memory_candidate',candidate_source)
    result={}
    for label,module in [('baseline',old),('candidate',new)]:
        module.torch.linspace(0.,1.,16,dtype=module.torch.float32)
        gc.collect();tracemalloc.start();tracemalloc.reset_peak()
        value=module.torch.linspace(0.,1.,1_000_000,dtype=module.torch.float32)
        current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        result[label]=dict(current_bytes=current,peak_bytes=peak,result_bytes=value.numel()*value.element_size())
        del value;gc.collect()
    return dict(source_hashes=dict(baseline=oh,candidate=nh),linspace_float32_n1000000=result,
                scope='CPU tracemalloc allocations during one warmed factory call with its result alive; not process RSS or WASM heap size.')


if __name__=='__main__':
    result=memory(sys.argv[1],sys.argv[2]) if '--memory' in sys.argv[3:] else run(sys.argv[1],sys.argv[2])
    print(json.dumps(result,indent=2))
