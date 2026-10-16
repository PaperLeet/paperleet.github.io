"""Real moments and reverse workloads; paired baseline uses separate operations."""
import argparse,hashlib,json,platform,statistics,time
from pathlib import Path
import numpy as np
import stridetorch
from stridetorch import torch


def run_benchmarks(*,dtype='float64',repeat=11,iterations=400,cases=None):
    if cases is not None:raise ValueError('Statistics controls use all thirteen fixed cases')
    dt=np.dtype(dtype);raw=(np.arange(4096,dtype=dt)/1024-1.5).reshape(32,128)
    small=torch.from_numpy(raw[0].copy());rows=torch.from_numpy(raw.copy())
    leaf=torch.from_numpy(raw[0].copy()).requires_grad_()
    fused='torch.var_mean' in torch._browser_capabilities
    def pair(x,kind):
        return getattr(torch,kind+'_mean')(x,dim=-1,correction=1) if fused else (getattr(x,kind)(dim=-1,correction=1),x.mean(-1))
    def reverse(kind,paired=False):
        leaf.grad=None
        output=pair(leaf,kind) if paired else (getattr(leaf,kind)(-1),)
        sum(v.sum() for v in output).backward()
    def common():
        leaf.grad=None;leaf.square().mean().backward()
    workloads={
        'var_small':lambda:small.var(-1),'std_small':lambda:small.std(-1),
        'var_mean_small':lambda:pair(small,'var'),'std_mean_small':lambda:pair(small,'std'),
        'var_rows':lambda:rows.var(-1),'std_rows':lambda:rows.std(-1),
        'var_mean_rows':lambda:pair(rows,'var'),'std_mean_rows':lambda:pair(rows,'std'),
        'var_reverse':lambda:reverse('var'),'std_reverse':lambda:reverse('std'),
        'var_mean_reverse':lambda:reverse('var',True),'std_mean_reverse':lambda:reverse('std',True),
        'common_reverse':common,
    }
    center=raw[0]-raw[0].mean();n=raw.shape[-1];variance=np.sum(center*center)/(n-1)
    for kind in ['var','std']:
        for paired in [False,True]:
            reverse(kind,paired);expected=2*center/(n-1)
            if kind=='std':expected=expected/(2*np.sqrt(variance))
            if paired:expected=expected+1/n
            np.testing.assert_allclose(leaf.grad.numpy(),expected,rtol=3e-5,atol=3e-7)
    for data in [small,rows]:
        a=data.numpy()
        for kind in ['var','std']:
            first,mean=pair(data,kind);expected=getattr(np,kind)(a,axis=-1,ddof=1)
            np.testing.assert_allclose(first.numpy(),expected,rtol=3e-5,atol=3e-7)
            np.testing.assert_allclose(mean.numpy(),a.mean(-1),rtol=3e-5,atol=3e-7)
    results={}
    for name,fn in workloads.items():
        count=max(1,iterations//10) if 'reverse' in name else max(1,iterations//4) if 'rows' in name else iterations
        for _ in range(3):fn()
        samples=[]
        for _ in range(repeat):
            start=time.perf_counter()
            for _ in range(count):fn()
            samples.append((time.perf_counter()-start)*1000/count)
        results[name]=dict(iterations=count,median_ms=statistics.median(samples),samples_ms=samples)
    root=Path(stridetorch.__file__).parent
    return dict(library=stridetorch.__version__,runtime_module=str(root/'_tensor.py'),python=platform.python_version(),numpy=np.__version__,dtype=dtype,repeat=repeat,iterations=iterations,warmup=3,paired_implementation='shared moments' if fused else 'separate variance/std and mean',scope='Thirteen fixed valid moments/reverse/control workloads. Baseline paired outputs compose existing operations;new paired outputs share computation. Excludes corrected scalar/empty/degenerate semantics and coldstart/application performance.',source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.suffix=='.py' or p.name in ('LICENSE','NOTICE')},results=results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dtype',default='float64');p.add_argument('--repeat',type=int,default=11);p.add_argument('--iterations',type=int,default=400);p.add_argument('--output');a=p.parse_args()
    result=run_benchmarks(dtype=a.dtype,repeat=a.repeat,iterations=a.iterations);raw=json.dumps(result,indent=2)+'\n'
    if a.output:Path(a.output).write_text(raw)
    print(raw)
