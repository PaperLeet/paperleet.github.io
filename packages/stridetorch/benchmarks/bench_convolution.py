"""Full Tensor convolution forward/backward: bounded auto dispatch vs direct.

Same library/source and unchanged mathematical objectives; not native Torch or
whole-application comparisons. Checks include every active gradient before time.
"""
import argparse,json,platform,statistics,sys,time
import numpy as np
from stridetorch import torch,__version__,_convolution as conv


def _fixture(shape,weight_shape,kwargs,kind,dtype):
    rng=np.random.default_rng(1913)
    x=torch.tensor(rng.normal(size=shape).astype(dtype),requires_grad=kind=='train')
    w=torch.tensor(rng.normal(size=weight_shape).astype(dtype),requires_grad=kind in ['train','weight'])
    b=torch.tensor(rng.normal(size=weight_shape[0]).astype(dtype),requires_grad=kind!='forward')
    dim=len(shape)-2;fn=getattr(torch.nn.functional,f'conv{dim}d')
    def operation():
        x.grad=w.grad=b.grad=None
        if kind=='forward':
            with torch.no_grad():return fn(x,w,b,**kwargs)
        out=fn(x,w,b,**kwargs);out.square().mean().backward();return out
    return operation,(x,w,b)


def run_benchmarks(*,repeat=9,iterations=20,warmup=3,dtype='float32',cases=None,implementation='auto'):
    if implementation not in ['auto','direct','matrix']:raise ValueError('Use auto/direct/matrix')
    if min(repeat,iterations)<1 or warmup<0:raise ValueError('Invalid sampling configuration')
    dtype=np.dtype(dtype)
    if dtype not in [np.dtype('float32'),np.dtype('float64')]:raise ValueError('Use float32/float64')
    definitions={
      'tiny_1d':((2,2,8),(4,2,3),dict(padding=1),'train'),
      'pointwise_2d':((4,8,16,16),(16,8,1,1),{},'train'),
      'spatial_2d':((2,8,16,16),(16,8,3,3),dict(padding=1),'train'),
      'depthwise_2d':((2,16,16,16),(32,1,3,3),dict(padding=1,groups=16),'train'),
      'dilated_1d':((4,8,128),(16,4,3),dict(padding=2,stride=2,dilation=2,groups=2),'train'),
      'volume_3d':((1,4,8,8,8),(8,4,3,3,3),dict(padding=1),'train'),
      'larger_2d':((2,16,32,32),(32,16,3,3),dict(padding=1),'train'),
      'spatial_forward':((2,8,16,16),(16,8,3,3),dict(padding=1),'forward'),
      'spatial_weight':((2,8,16,16),(16,8,3,3),dict(padding=1),'weight'),
      'spatial_bias':((2,8,16,16),(16,8,3,3),dict(padding=1),'bias')}
    names=list(definitions) if cases is None else list(cases)
    if not names or len(set(names))!=len(names) or any(n not in definitions for n in names):raise ValueError('Unknown/duplicate case')
    reports={};checks={};previous=conv._ALGORITHM
    try:
        for name in names:
            shape,weight_shape,kwargs,kind=definitions[name];op,inputs=_fixture(shape,weight_shape,kwargs,kind,dtype)
            conv._ALGORITHM='direct';a=op();reference=[a.numpy().copy(),*[t.grad.numpy().copy() for t in inputs if t.grad is not None]]
            conv._ALGORITHM=implementation;b=op();actual=[b.numpy(),*[t.grad.numpy() for t in inputs if t.grad is not None]]
            assert len(actual)==len(reference)
            tolerance=6e-5 if dtype==np.dtype('float32') else 2e-11
            for x,y in zip(actual,reference):np.testing.assert_allclose(x,y,rtol=tolerance,atol=2e-5 if dtype==np.dtype('float32') else 3e-12)
            checks[name]=dict(max_abs_errors=[float(np.max(np.abs(x-y))) for x,y in zip(actual,reference)])
            geometry=conv._validate(*inputs,kwargs.get('stride',1),kwargs.get('padding',0),kwargs.get('dilation',1),kwargs.get('groups',1),len(shape)-2,'zeros')
            choice=conv._use_matrix(inputs[0],inputs[1],geometry)
            workspace=conv._matrix_workspace(geometry[0],weight_shape,geometry[3],geometry[7],geometry[9],dtype.itemsize)
            for _ in range(warmup):op()
            count=iterations*4 if name.startswith('tiny') else max(1,iterations//8) if name.startswith('larger') else iterations
            samples=[]
            for _ in range(repeat):
                start=time.perf_counter()
                for __ in range(count):op()
                samples.append((time.perf_counter()-start)*1000/count)
            reports[name]=dict(shape=shape,weight_shape=weight_shape,kwargs=kwargs,kind=kind,iterations=count,samples_ms=samples,median_ms=statistics.median(samples),selected='matrix' if choice else 'direct',estimated_matrix_numpy_workspace_bytes=workspace)
    finally:conv._ALGORITHM=previous
    return dict(library_version=__version__,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),runtime_module=sys.modules[torch.Tensor.__module__].__file__,implementation=implementation,dtype=dtype.name,repeat=repeat,iterations=iterations,warmup=warmup,seed=1913,reference_checks=checks,results=reports,scope=__doc__)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--repeat',type=int,default=9);p.add_argument('--iterations',type=int,default=20);p.add_argument('--dtype',choices=['float32','float64'],default='float32');p.add_argument('--implementation',choices=['auto','direct','matrix'],default='auto');p.add_argument('--case',action='append',dest='cases');a=p.parse_args()
    raw=json.dumps(run_benchmarks(repeat=a.repeat,iterations=a.iterations,dtype=a.dtype,cases=a.cases,implementation=a.implementation),indent=2)+'\n'
    if a.output:open(a.output,'w').write(raw)
    print(raw)
