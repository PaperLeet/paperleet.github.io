"""Classification kernel timings with independent first-order value/VJP checks."""
import argparse,json,platform,statistics,sys,time
import numpy as np
from torchwasm import torch,__version__


def _fixture(shape,kind,dtype,implementation):
    rng=np.random.default_rng(1812);classes=shape[1];axes=(0,*range(2,len(shape)));dim=1
    raw=rng.normal(size=shape).astype(dtype);labels=rng.integers(0,classes,size=(shape[0],*shape[2:]),dtype=np.int64)
    labels.reshape(-1)[0]=-100
    soft=kind=='soft';smoothed=kind=='smooth';weighted=kind in ('smooth','soft','nll')
    target=rng.uniform(.1,1,size=shape).astype(dtype) if soft else labels
    weights=rng.uniform(.2,2,size=classes).astype(dtype) if weighted else None
    x=torch.tensor(raw,requires_grad=True);y=torch.tensor(target,requires_grad=soft);w=torch.tensor(weights,requires_grad=soft) if weighted else None
    broadcast=(1,classes,*([1]*(len(shape)-2)));s=.15 if smoothed else 0.
    def forward():
        if implementation=='joint':
            fn=torch.nn.functional.nll_loss if kind=='nll' else torch.nn.functional.cross_entropy
            return fn(x,y,w,**({'label_smoothing':s} if kind!='nll' else {}))
        lp=x if kind=='nll' else x.log_softmax(1)
        if soft:return -(lp*y*w.reshape(broadcast)).sum(1).mean()
        valid=labels!=-100;safe=np.where(valid,labels,0)
        selected=lp.gather(1,torch.tensor(safe).unsqueeze(1)).squeeze(1)
        selected_weight=torch.tensor(np.where(valid,weights[safe] if weighted else 1,0).astype(dtype))
        denominator=selected_weight.sum()
        base=-(selected*selected_weight).sum()/denominator
        if not s:return base
        smooth=-(lp*w.reshape(broadcast)).sum(1)*torch.tensor(valid.astype(dtype))
        return (1-s)*base+s/classes*smooth.sum()/denominator
    def operation():
        x.grad=None;y.grad=None
        if w is not None:w.grad=None
        value=forward();value.backward();return value
    # Independent float64 formulas; no second adapter path supplies the oracle.
    z=raw.astype(np.float64);shifted=z-z.max(1,keepdims=True);lp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True)) if kind!='nll' else z
    if soft:
        effective=target.astype(np.float64)*weights.reshape(broadcast);denominator=labels.size
        expected_value=-(lp*effective).sum()/denominator
        dx=(np.exp(lp)*effective.sum(1,keepdims=True)-effective)/denominator
        dy=-lp*weights.reshape(broadcast)/denominator;dw=-(lp*target).sum(axis=axes)/denominator
    else:
        valid=labels!=-100;safe=np.where(valid,labels,0);selected_weights=np.where(valid,weights[safe] if weighted else 1,0)
        denominator=selected_weights.sum();coeff=np.zeros(shape,dtype=np.float64)
        np.put_along_axis(coeff,np.expand_dims(safe,1),np.expand_dims(selected_weights*(1-s),1),axis=1)
        if s:coeff+=s/classes*weights.reshape(broadcast)*np.expand_dims(valid,1)
        expected_value=-(lp*coeff).sum()/denominator
        dx=(-coeff if kind=='nll' else np.exp(lp)*coeff.sum(1,keepdims=True)-coeff)/denominator
        dy=dw=None
    value=operation();pairs=[(value,np.asarray(expected_value)),(x.grad,dx)]
    if soft:pairs.extend([(y.grad,dy),(w.grad,dw)])
    tolerance=2e-5 if dtype==np.dtype('float32') else 5e-12
    for actual,expected in pairs:np.testing.assert_allclose(actual.numpy(),expected,rtol=tolerance,atol=tolerance)
    checks=[float(np.max(np.abs(a.numpy()-b))) for a,b in pairs]
    return operation,checks


def run_benchmarks(*,repeat=9,iterations=80,warmup=3,dtype='float32',cases=None,implementation='joint'):
    if implementation not in ('joint','composite'):raise ValueError('Use joint or composite')
    dtype=np.dtype(dtype)
    if dtype not in (np.dtype('float32'),np.dtype('float64')) or min(repeat,iterations)<1 or warmup<0:raise ValueError('Invalid benchmark configuration')
    definitions={'tiny_index':((8,16),'index'),'classifier_index':((256,128),'index'),'vocabulary_index':((32,4096),'index'),
                 'spatial_smooth':((8,16,16,16),'smooth'),'probability_trainable':((128,64),'soft'),'spatial_nll':((8,16,16,16),'nll')}
    names=list(definitions) if cases is None else list(cases)
    if not names or len(set(names))!=len(names) or any(name not in definitions for name in names):raise ValueError('Unknown/duplicate case')
    results={};checks={}
    for name in names:
        shape,kind=definitions[name];operation,checks[name]=_fixture(shape,kind,dtype,implementation)
        for _ in range(warmup):operation()
        count=iterations*4 if name.startswith('tiny') else max(1,iterations//4) if name.startswith(('vocabulary','spatial')) else iterations
        samples=[]
        for _ in range(repeat):
            start=time.perf_counter()
            for __ in range(count):operation()
            samples.append((time.perf_counter()-start)*1000/count)
        results[name]=dict(shape=shape,kind=kind,iterations=count,samples_ms=samples,median_ms=statistics.median(samples))
    return dict(library_version=__version__,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                runtime_module=sys.modules[torch.Tensor.__module__].__file__,implementation=implementation,dtype=dtype.name,repeat=repeat,iterations=iterations,warmup=warmup,seed=1812,reference_checks=checks,results=results,
                scope='First-order classification kernels. NumPy-only checks run before timing. Same-source joint/composition comparisons and old-release common-case comparisons require separately labeled balanced process series; all samples retained.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--repeat',type=int,default=9);p.add_argument('--iterations',type=int,default=80);p.add_argument('--dtype',choices=['float32','float64'],default='float32');p.add_argument('--implementation',choices=['joint','composite'],default='joint');p.add_argument('--case',action='append',dest='cases');a=p.parse_args()
    raw=json.dumps(run_benchmarks(repeat=a.repeat,iterations=a.iterations,dtype=a.dtype,cases=a.cases,implementation=a.implementation),indent=2)+'\n'
    if a.output:
        with open(a.output,'w') as f:f.write(raw)
    print(raw)
