"""BatchNorm joint NumPy kernels versus a Tensor-composition reference.

Both implementations use this library. The reference covers first-order
normalization, not native saved-statistic JVP tape conventions. This is not a
comparison to native PyTorch or a previous release without BatchNorm support.
"""
import argparse
import json
import platform
import statistics
import sys
import time
import numpy as np
from torchwasm import __version__, torch


def _fixture(shape, kind, dtype, implementation):
    rng=np.random.default_rng(1616)
    x=torch.tensor(rng.normal(size=shape).astype(dtype),requires_grad=kind in ('train','eval'))
    w=torch.tensor(rng.uniform(.7,1.3,size=shape[1]).astype(dtype),requires_grad=kind not in ('bias','forward'))
    b=torch.tensor(rng.normal(size=shape[1]).astype(dtype),requires_grad=kind!='forward')
    mean=torch.zeros(shape[1],dtype=dtype);variance=torch.ones(shape[1],dtype=dtype)
    axes=(0,*range(2,len(shape)));broadcast=(1,shape[1],*([1]*(len(shape)-2)))
    training=kind!='eval'
    def forward():
        if implementation=='joint':
            return torch.nn.functional.batch_norm(x,None if training else mean,None if training else variance,w,b,training=training)
        if training:
            centered=x-x.mean(axes,True)
            inverse=(centered.square().mean(axes,True)+1e-5).rsqrt()
        else:
            centered=x-mean.reshape(broadcast)
            inverse=(variance.reshape(broadcast)+1e-5).rsqrt()
        return centered*inverse*w.reshape(broadcast)+b.reshape(broadcast)
    def operation():
        x.grad=w.grad=b.grad=None
        if kind=='forward':
            with torch.no_grad():return forward()
        output=forward();output.square().mean().backward();return output
    return operation,(x,w,b)


def run_benchmarks(*,repeat=11,iterations=100,warmup=3,dtype='float32',cases=None,implementation='joint'):
    if implementation not in ('joint','composite'):raise ValueError('Use joint or composite')
    dtype=np.dtype(dtype)
    if dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Use float32/float64')
    if min(repeat,iterations)<1 or warmup<0:raise ValueError('Invalid sampling configuration')
    definitions={'tiny_train':((8,16),'train'),'mlp_train':((256,128),'train'),
                 'vision_train':((8,16,16,16),'train'),'mlp_eval':((256,128),'eval'),
                 'mlp_affine_only':((256,128),'affine'),'mlp_bias_only':((256,128),'bias'),
                 'mlp_forward':((256,128),'forward')}
    names=list(definitions) if cases is None else list(cases)
    if not names or len(set(names))!=len(names) or any(name not in definitions for name in names):raise ValueError('Unknown or duplicate case')
    results={};checks={}
    for name in names:
        shape,kind=definitions[name]
        left,inputs=_fixture(shape,kind,dtype,'joint');right,reference=_fixture(shape,kind,dtype,'composite')
        a,b=left(),right();pairs=[(a,b)]+[(x.grad,y.grad) for x,y in zip(inputs,reference) if x.grad is not None]
        tolerance=3e-5 if dtype==np.dtype('float32') else 3e-12
        for x,y in pairs:np.testing.assert_allclose(x.numpy(),y.numpy(),rtol=tolerance,atol=tolerance)
        checks[name]=dict(max_abs_errors=[float(np.max(np.abs(x.numpy()-y.numpy()))) for x,y in pairs])
        operation=left if implementation=='joint' else right
        for _ in range(warmup):operation()
        count=iterations*4 if name.startswith('tiny') else max(1,iterations//8) if name.startswith('vision') else iterations
        samples=[]
        for _ in range(repeat):
            start=time.perf_counter()
            for __ in range(count):operation()
            samples.append((time.perf_counter()-start)*1000/count)
        results[name]=dict(shape=shape,kind=kind,iterations=count,samples_ms=samples,median_ms=statistics.median(samples))
    return dict(library_version=__version__,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                runtime_module=sys.modules[torch.Tensor.__module__].__file__,implementation=implementation,dtype=dtype.name,
                repeat=repeat,iterations=iterations,warmup=warmup,seed=1616,reference_checks=checks,results=results,
                scope='First-order joint BatchNorm versus mathematically equivalent Tensor composition in this same library. Import/fixtures excluded; no native or old-release comparison; all samples retained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output');parser.add_argument('--repeat',type=int,default=11)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--dtype',choices=['float32','float64'],default='float32')
    parser.add_argument('--implementation',choices=['joint','composite'],default='joint')
    parser.add_argument('--case',action='append',dest='cases')
    options=parser.parse_args()
    report=json.dumps(run_benchmarks(repeat=options.repeat,iterations=options.iterations,dtype=options.dtype,cases=options.cases,implementation=options.implementation),indent=2)+'\n'
    if options.output:
        with open(options.output,'w',encoding='utf8') as stream:stream.write(report)
    print(report)
