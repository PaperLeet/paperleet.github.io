"""Matching valid gather/sort/topk work; corrected half/invalid cases excluded."""
import argparse,json,platform,statistics,time
from pathlib import Path
import numpy as np
import stridetorch
from stridetorch import torch


def run_benchmarks(*,dtype='float32',repeat=11,iterations=500,cases=None):
    if cases is not None:raise ValueError('Indexing controls use all fixed cases')
    dt=np.dtype(dtype)
    raw=((np.arange(4*8192,dtype=np.int64)*37)%65537).astype(dt).reshape(4,8192)/32768
    rows=torch.from_numpy(raw.copy());columns=torch.from_numpy(raw.T)
    small=torch.from_numpy(raw[0,:128].copy());tracked=torch.from_numpy(raw.copy()).requires_grad_()
    small_index=torch.tensor(np.arange(32)%13,dtype=torch.int64)
    row_index=torch.tensor(np.tile(np.arange(128)%31,(4,1)),dtype=torch.int64)
    def reverse(kind):
        tracked.grad=None
        out=tracked.gather(1,row_index) if kind=='gather' else tracked.sort(dim=1).values if kind=='sort' else tracked.topk(8,dim=1).values
        out.sum().backward()
    workloads={
        'gather_small':lambda:small.gather(0,small_index),
        'gather_rows':lambda:rows.gather(1,row_index),
        'sort_small':lambda:small.sort(),
        'topk_small':lambda:small.topk(8),
        'topk_rows8':lambda:rows.topk(8,dim=1),
        'topk_columns8':lambda:columns.topk(8,dim=0),
        'topk_unsorted_rows8':lambda:rows.topk(8,dim=1,sorted=False),
        'topk_dense_rows':lambda:rows.topk(4096,dim=1),
        'gather_rows_reverse':lambda:reverse('gather'),
        'topk_rows_reverse':lambda:reverse('topk'),
        'sort_rows_reverse':lambda:reverse('sort'),
        'control_exp_small':lambda:small.exp(),
    }
    np.testing.assert_array_equal(workloads['gather_small']().numpy(),raw[0,:128][small_index.numpy()])
    np.testing.assert_array_equal(workloads['gather_rows']().numpy(),np.take_along_axis(raw,row_index.numpy(),axis=1))
    np.testing.assert_array_equal(workloads['sort_small']().values.numpy(),np.sort(raw[0,:128]))
    for name in ('topk_rows8','topk_unsorted_rows8'):
        np.testing.assert_array_equal(np.sort(workloads[name]().values.numpy(),axis=1),np.sort(raw,axis=1)[:,-8:])
    np.testing.assert_array_equal(workloads['topk_columns8']().values.numpy(),np.sort(raw.T,axis=0)[-8:][::-1])
    for kind in ('gather','sort','topk'):
        reverse(kind);expected=np.zeros_like(raw)
        if kind=='gather':
            for row in range(4):np.add.at(expected[row],row_index.numpy()[row],1)
        elif kind=='sort':expected.fill(1)
        else:np.put_along_axis(expected,np.argsort(raw,axis=1)[:,-8:],1,axis=1)
        np.testing.assert_array_equal(tracked.grad.numpy(),expected)
    results={}
    for name,fn in workloads.items():
        count=max(1,iterations//20) if 'rows' in name or 'columns' in name else iterations
        for _ in range(3):fn()
        samples=[]
        for _ in range(repeat):
            start=time.perf_counter()
            for _ in range(count):fn()
            samples.append((time.perf_counter()-start)*1000/count)
        results[name]=dict(iterations=count,median_ms=statistics.median(samples),samples_ms=samples)
    return dict(library=stridetorch.__version__,runtime_module=str(Path(stridetorch.__file__).with_name('_tensor.py')),python=platform.python_version(),numpy=np.__version__,dtype=dtype,repeat=repeat,iterations=iterations,warmup=3,scope='Twelve identical valid gather/sort/topk method workloads and exp control;large rows4x8192,columns8192x4,k8/k4096,finite unique values and exact gradients. Excludes new functional API,JVP,half accuracy fixes,invalid schemas,coldstart,native Torch,peak memory and application speed.',results=results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dtype',default='float32');p.add_argument('--repeat',type=int,default=11);p.add_argument('--iterations',type=int,default=500);p.add_argument('--output');a=p.parse_args()
    raw=json.dumps(run_benchmarks(dtype=a.dtype,repeat=a.repeat,iterations=a.iterations),indent=2)+'\n'
    if a.output:Path(a.output).write_text(raw)
    print(raw,end='')
