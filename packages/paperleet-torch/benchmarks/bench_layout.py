"""Unchanged valid view workloads and common Tensor-allocation regression controls."""
import argparse,gc,hashlib,json,platform,statistics,time
from pathlib import Path
import numpy as np
import paperleet_torch
from paperleet_torch import torch


def run_benchmarks(*,dtype='float64',repeat=11,iterations=1200):
    dt=np.dtype(dtype);a=np.arange(64,dtype=dt)/64
    contiguous=torch.from_numpy(a);stepped=torch.from_numpy(np.repeat(a,2)[::2])
    transposed=torch.from_numpy(np.arange(64,dtype=dt).reshape(8,8).T)
    x=torch.from_numpy(a.copy()).requires_grad_();t=torch.ones(64,dtype=dt)
    source=np.arange(8192,dtype=dt).reshape(64,128)
    def reverse():
        x.grad=None;x.square().mean().backward()
    def view_reverse():
        x.grad=None;x.view(8,8).square().mean().backward()
    def view_jvp():return torch.func.jvp(lambda z:z.view(8,8),(contiguous,),(t,))
    cases={
      'scalar_tensor':lambda:torch.tensor(.3),
      'array_import':lambda:torch.from_numpy(source),
      'array_copy':lambda:torch.tensor(source),
      'view_contiguous':lambda:contiguous.view(8,8),
      'view_stepped':lambda:stepped.view(8,8),
      'view_transposed_shape':lambda:transposed.view(8,8),
      'common_reverse':reverse,
      'view_reverse':view_reverse,
      'view_jvp_matching':view_jvp,
    }
    for owner in [contiguous,stepped,transposed]:
        y=owner.view(8,8)
        assert y.data_ptr()==owner.data_ptr()
        np.testing.assert_array_equal(y.numpy(),owner.numpy().reshape(8,8))
    expected=2*a/a.size
    for fn in [reverse,view_reverse]:
        fn();np.testing.assert_allclose(x.grad.numpy(),expected,rtol=2e-6,atol=2e-7)
    p,q=view_jvp();np.testing.assert_array_equal(p.numpy(),a.reshape(8,8));np.testing.assert_array_equal(q.numpy(),np.ones((8,8)))
    results={}
    for name,fn in cases.items():
        count=max(1,iterations//10) if 'reverse' in name or 'jvp' in name else iterations
        for _ in range(3):fn()
        samples=[]
        for _ in range(repeat):
            start=time.perf_counter()
            for _ in range(count):fn()
            samples.append((time.perf_counter()-start)*1000/count)
        results[name]=dict(iterations=count,median_ms=statistics.median(samples),samples_ms=samples)
    root=Path(paperleet_torch.__file__).parent
    return dict(library=paperleet_torch.__version__,runtime_module=str(root/'_tensor.py'),python=platform.python_version(),numpy=np.__version__,dtype=dtype,repeat=repeat,iterations=iterations,warmup=3,scope='Same successful view/alias/value/derivative workloads plus common allocation controls. Does not time newly rejected views or mismatched-layout JVPs;does not compare equivalent full-native Torch performance.',source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.suffix=='.py' or p.name in ('LICENSE','NOTICE')},results=results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dtype',default='float64');p.add_argument('--repeat',type=int,default=11);p.add_argument('--iterations',type=int,default=1200);p.add_argument('--output');a=p.parse_args()
    result=run_benchmarks(dtype=a.dtype,repeat=a.repeat,iterations=a.iterations);raw=json.dumps(result,indent=2)+'\n'
    if a.output:Path(a.output).write_text(raw)
    print(raw)
