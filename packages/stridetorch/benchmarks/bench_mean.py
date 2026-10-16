"""Matching valid mean workloads; excludes repaired overflow/invalid overloads."""
import argparse,json,platform,statistics,time
from pathlib import Path
import numpy as np
import stridetorch
from stridetorch import torch


def run_benchmarks(*,dtype='float32',repeat=11,iterations=1000,cases=None):
    if cases is not None:raise ValueError('Mean controls use all twelve fixed cases')
    dt=np.dtype(dtype);raw=(np.arange(4096,dtype=dt)/1024-1.5).reshape(32,128)
    small=torch.from_numpy(raw[0].copy());rows=torch.from_numpy(raw.copy())
    half=torch.from_numpy(np.ones((32,128),dtype=np.float16))
    leaf=torch.from_numpy(raw[0].copy()).requires_grad_()
    tracked=torch.from_numpy(raw.copy()).requires_grad_()
    target=np.float64 if dt==np.float32 else np.float32
    def reverse(functional=False,cast=False):
        leaf.grad=None
        output=torch.mean(leaf) if functional else leaf.mean(dtype=target) if cast else leaf.mean()
        output.backward()
    def reverse_rows():
        tracked.grad=None;torch.mean(tracked,dim=-1).sum().backward()
    workloads={
        'mean_small_method':lambda:small.mean(),
        'mean_small_function':lambda:torch.mean(small),
        'mean_rows_method':lambda:rows.mean(-1),
        'mean_rows_function':lambda:torch.mean(rows,dim=-1),
        'mean_columns_method':lambda:rows.mean(0),
        'mean_half_rows':lambda:half.mean(-1),
        'mean_cast_method':lambda:small.mean(dtype=target),
        'mean_reverse_method':lambda:reverse(),
        'mean_reverse_function':lambda:reverse(True),
        'mean_cast_reverse':lambda:reverse(cast=True),
        'mean_rows_reverse':reverse_rows,
        'control_exp_method':lambda:small.exp(),
    }
    for name in ('mean_small_method','mean_small_function','mean_cast_method'):
        np.testing.assert_allclose(workloads[name]().numpy(),raw[0].mean(),rtol=2e-6)
    for name in ('mean_rows_method','mean_rows_function'):
        np.testing.assert_allclose(workloads[name]().numpy(),raw.mean(-1),rtol=2e-6)
    np.testing.assert_allclose(workloads['mean_columns_method']().numpy(),raw.mean(0),rtol=2e-6,atol=1e-6)
    np.testing.assert_array_equal(workloads['mean_half_rows']().numpy(),np.ones(32,dtype=np.float16))
    for name in ('mean_reverse_method','mean_reverse_function','mean_cast_reverse'):
        workloads[name]();np.testing.assert_array_equal(leaf.grad.numpy(),np.full(128,1/128,dtype=dt))
    reverse_rows();np.testing.assert_array_equal(tracked.grad.numpy(),np.full((32,128),1/128,dtype=dt))
    np.testing.assert_allclose(workloads['control_exp_method']().numpy(),np.exp(raw[0]),rtol=2e-6)
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
    return dict(library=stridetorch.__version__,runtime_module=str(Path(stridetorch.__file__).with_name('_tensor.py')),python=platform.python_version(),numpy=np.__version__,dtype=dtype,repeat=repeat,iterations=iterations,warmup=3,scope='Twelve identical valid CPU/WASM means,reverse and unchanged exp control;baseline already supports method dtype. Excludes repaired overflow/invalid overloads,unsupported baseline functional dtype,coldstart/native Torch/application speed.',results=results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dtype',default='float32');p.add_argument('--repeat',type=int,default=11);p.add_argument('--iterations',type=int,default=1000);p.add_argument('--output');a=p.parse_args()
    raw=json.dumps(run_benchmarks(dtype=a.dtype,repeat=a.repeat,iterations=a.iterations),indent=2)+'\n'
    if a.output:Path(a.output).write_text(raw)
    print(raw,end='')
