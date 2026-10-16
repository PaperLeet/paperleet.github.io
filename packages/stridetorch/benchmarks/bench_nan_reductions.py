"""Valid matching finite-gradient NaN reductions versus a where/sum composite."""
import argparse,json,platform,statistics,time
from pathlib import Path
import numpy as np
import stridetorch
from stridetorch import torch


def run_benchmarks(*,dtype='float32',repeat=11,iterations=1000,cases=None,implementation='direct'):
    if cases is not None:raise ValueError('NaN controls use all fixed cases')
    if implementation not in ('direct','composite'):raise ValueError('Use direct or composite')
    dt=np.dtype(dtype);raw=(np.arange(4096,dtype=dt)/1024-1.5).reshape(32,128)
    raw.flat[::7]=np.nan
    small=torch.from_numpy(raw[0].copy());rows=torch.from_numpy(raw.copy())
    tracked=torch.from_numpy(raw.copy()).requires_grad_()
    target=np.float64 if dt==np.float32 else np.float32
    def reduce(x,mean=False,dim=None,dtype=None):
        if implementation=='direct':return (torch.nanmean if mean else torch.nansum)(x,dim=dim,dtype=dtype)
        valid=~torch.isnan(x)
        value=torch.where(valid,x,0.).sum(dim,dtype=dtype)
        return value/valid.sum(dim) if mean else value
    def reverse(mean):
        tracked.grad=None;reduce(tracked,mean,dim=-1).sum().backward()
    workloads={
        'nansum_small':lambda:reduce(small),
        'nanmean_small':lambda:reduce(small,True),
        'nansum_rows':lambda:reduce(rows,dim=-1),
        'nanmean_rows':lambda:reduce(rows,True,dim=-1),
        'nanmean_columns':lambda:reduce(rows,True,dim=0),
        'nanmean_cast':lambda:reduce(small,True,dtype=target),
        'nansum_rows_reverse':lambda:reverse(False),
        'nanmean_rows_reverse':lambda:reverse(True),
        'control_sum_small':lambda:small.sum(),
        'control_mean_small':lambda:small.mean(),
        'control_exp_small':lambda:small.exp(),
    }
    np.testing.assert_allclose(workloads['nansum_small']().numpy(),np.nansum(raw[0]),rtol=3e-6)
    np.testing.assert_allclose(workloads['nanmean_small']().numpy(),np.nanmean(raw[0]),rtol=3e-6)
    np.testing.assert_allclose(workloads['nansum_rows']().numpy(),np.nansum(raw,axis=-1),rtol=3e-6)
    for key in ('nanmean_rows','nanmean_columns'):
        np.testing.assert_allclose(workloads[key]().numpy(),np.nanmean(raw,axis=-1 if key.endswith('rows') else 0),rtol=3e-6,atol=1e-7)
    for mean in (False,True):
        reverse(mean)
        expected=np.where(np.isnan(raw),0,1/(np.sum(~np.isnan(raw),axis=-1,keepdims=True) if mean else 1))
        np.testing.assert_allclose(tracked.grad.numpy(),expected,rtol=3e-6)
    results={}
    for name,fn in workloads.items():
        count=max(1,iterations//10) if 'reverse' in name else max(1,iterations//4) if 'rows' in name or 'columns' in name else iterations
        for _ in range(3):fn()
        samples=[]
        for _ in range(repeat):
            start=time.perf_counter()
            for _ in range(count):fn()
            samples.append((time.perf_counter()-start)*1000/count)
        results[name]=dict(iterations=count,median_ms=statistics.median(samples),samples_ms=samples)
    return dict(library=stridetorch.__version__,runtime_module=str(Path(stridetorch.__file__).with_name('_tensor.py')),python=platform.python_version(),numpy=np.__version__,dtype=dtype,implementation=implementation,repeat=repeat,iterations=iterations,warmup=3,scope='Matching finite-gradient NaN reductions with at least one valid entry per slice; baseline where/sum composite versus direct kernels. Includes unchanged ordinary controls. No all-NaN/IEEE-gradient,half,coldstart,native Torch,peak-memory or app speed claims.',results=results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dtype',default='float32');p.add_argument('--repeat',type=int,default=11);p.add_argument('--iterations',type=int,default=1000);p.add_argument('--implementation',choices=['direct','composite'],default='direct');p.add_argument('--output');a=p.parse_args()
    raw=json.dumps(run_benchmarks(dtype=a.dtype,repeat=a.repeat,iterations=a.iterations,implementation=a.implementation),indent=2)+'\n'
    if a.output:Path(a.output).write_text(raw)
    print(raw,end='')
