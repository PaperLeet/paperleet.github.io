"""Balanced warm public Tensor/SDPA/PaperLeet comparisons, with A/A controls."""
import argparse,gc,hashlib,importlib.util,json,math,random,statistics,sys,time
from pathlib import Path
import numpy as np


def load(root,label):
    name='_batch_benchmark_'+label
    spec=importlib.util.spec_from_file_location(name,str(Path(root)/'__init__.py'),submodule_search_locations=[str(root)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def specifications():
    rows=[]
    for dtype in ['float32','float64']:
        for name,a,b,layout,backward in [
            ('small_cores',(32,32,64),(32,64,32),'C',False),
            ('scores64',(2,4,64,64),(2,4,64,64),'attention',False),
            ('scores128',(2,4,128,64),(2,4,64,128),'attention',True),
            ('scores256',(1,4,256,64),(1,4,64,256),'attention',False),
            ('weights_v',(2,4,128,128),(2,4,128,64),'C',True),
            ('left_broadcast',(1,128,64),(8,64,128),'C',True),
            ('right_broadcast',(8,128,64),(64,128),'C',True),
            ('cross_broadcast',(2,1,128,64),(1,3,64,128),'C',True),
            ('fortran',(4,128,128),(4,128,128),'F',True),
            ('stride',(4,128,128),(4,128,128),'stride',True),
            ('long_reduce',(8,32,1024),(8,1024,32),'C',False),
            ('repeat_batch',(8,64,64),(8,64,64),'repeat',True),
            ('tiny2d',(4,5),(5,6),'C',False),
            ('large2d',(256,256),(256,256),'C',True),
        ]:
            for mode in ['forward','backward'] if backward else ['forward']:
                rows.append(dict(key=f'{dtype}_{name}_{mode}',dtype=dtype,kind='matmul',name=name,shapes=[a,b],layout=layout,mode=mode))
        for name,length,order,grouped,mask in [('sdpa64',64,'C',False,False),('sdpa128',128,'attention',False,False),('gqa128',128,'attention',True,True),('sdpa_tiny',8,'C',False,False),('paper_attention',128,'attention',False,True),('paper_gqa',128,'attention',True,True)]:
            features=8 if name=='sdpa_tiny' else 64
            shapes=[(2,4,length,features),(2,2 if grouped else 4,length,features),(2,2 if grouped else 4,length,features)]
            for mode in ['forward','backward'] if name.startswith('paper_') else ['forward','backward','jvp']:
                rows.append(dict(key=f'{dtype}_{name}_{mode}',dtype=dtype,kind='sdpa',name=name,shapes=shapes,layout=order,grouped=grouped,mask=mask,mode=mode))
    for dtype in ['float32','float64']:
        for name, a, b, backward in [
            ('dot8', (8,), (8,), True),
            ('mv16', (16,32), (32,), True),
            ('vector_batch', (16,), (4,16,8), True),
            ('empty_mv', (0,512), (512,), True),
            ('empty_broadcast', (8,1,0,32), (1,4,32,64), True)]:
            for mode in ['forward','backward'] if backward else ['forward']:
                rows.append(dict(key=f'{dtype}_{name}_{mode}',dtype=dtype,kind='matmul',name=name,shapes=[a,b],layout='C',mode=mode))
    return rows


def arrange(value,order,transpose=False):
    if order=='F':return np.asfortranarray(value)
    if order=='stride':return np.repeat(value,2,axis=-1)[...,::2]
    if order=='repeat':return np.broadcast_to(value[:1],value.shape)
    if order=='attention':
        if transpose:value=value.swapaxes(-1,-2)
        value=np.ascontiguousarray(value.swapaxes(-3,-2)).swapaxes(-3,-2)
        return value.swapaxes(-1,-2) if transpose else value
    return value


def make_inputs(spec):
    rng=np.random.default_rng(77181)
    inputs=[arrange(rng.integers(-32,33,size=shape,dtype=np.int32).astype(spec['dtype'])/128,spec['layout'],spec['kind']=='matmul' and index==1 and spec['layout']=='attention') for index,shape in enumerate(spec['shapes'])]
    tangents=[arrange(rng.integers(-16,17,size=shape,dtype=np.int32).astype(spec['dtype'])/128,spec['layout'],spec['kind']=='matmul' and index==1 and spec['layout']=='attention') for index,shape in enumerate(spec['shapes'])]
    if spec['kind']=='matmul':shape=np.shape(inputs[0] @ inputs[1])
    else:shape=inputs[0].shape[:-1]+(inputs[2].shape[-1],)
    seed=rng.integers(-16,17,size=shape,dtype=np.int32).astype(spec['dtype'])/128
    seed=np.asarray(seed,dtype=spec['dtype'])
    mask=np.tril(np.ones((shape[-2],shape[-2]),dtype=bool)) if spec['kind']!='matmul' else None
    if mask is not None:mask[0]=False
    return inputs,tangents,seed,mask


def paper_functions(module,fixtures):
    result={}
    for name,filename,attribute in [('paper_attention','paperleet_attention.py','scaled_dot_product_attention'),('paper_gqa','paperleet_gqa.py','grouped_query_attention')]:
        namespace={'__name__':'_paper_benchmark_'+name}
        with module.as_torch():exec(compile((Path(fixtures)/filename).read_text(),filename,'exec'),namespace)
        result[name]=namespace[attribute]
    return result


def make_flow(module,spec,raw,paper):
    api=module.torch;values,directions,seed,mask=raw
    inputs=tuple(api.from_numpy(x).requires_grad_(spec['mode']=='backward') for x in values)
    tangents=tuple(api.from_numpy(x) for x in directions);seed=api.from_numpy(seed);mask=api.from_numpy(mask) if spec.get('mask') else None
    if spec['kind']=='matmul':function=(lambda a,b:api.dot(a,b)) if spec['name']=='dot8' else (lambda a,b:a@b)
    elif spec['name'].startswith('paper_'):
        source=paper[spec['name']];function=lambda q,k,v:source(q,k,v,mask)[0]
    else:function=lambda q,k,v:api.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=spec['grouped'])
    if spec['mode']=='forward':return lambda:(function(*inputs),)
    if spec['mode']=='jvp':return lambda:api.func.jvp(function,inputs,tangents)
    def backward():
        y=function(*inputs);return (y,*api.autograd.grad(y,inputs,seed))
    return backward


def elapsed(fn,count):
    start=time.perf_counter()
    for _ in range(count):fn()
    return time.perf_counter()-start


def calibrate(fn):
    count=1
    while True:
        seconds=elapsed(fn,count)
        if seconds>=.02:return max(1,math.ceil(count*.1/seconds))
        count*=max(2,min(32,math.ceil(.025/max(seconds,1e-9))))


def run(baseline,candidate,fixtures,cycle=0,sample_count=3):
    modules={name:load(root,name) for name,root in [('A0',baseline),('A1',baseline),('B',candidate)]}
    paper={name:paper_functions(module,fixtures) for name,module in modules.items()}
    source={name:{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in Path(root).iterdir() if q.is_file()} for name,root in [('baseline',baseline),('candidate',candidate)]}
    specs=specifications();random.Random(77180+cycle).shuffle(specs);rows=[]
    for index,spec in enumerate(specs):
        raw=make_inputs(spec);functions={name:make_flow(module,spec,raw,paper[name]) for name,module in modules.items()}
        outputs={name:fn() for name,fn in functions.items()}
        for name,output in outputs.items():
            for actual,expected in zip(output,outputs['A0']):
                np.testing.assert_allclose(actual.detach().numpy(),expected.detach().numpy(),rtol=4e-4,atol=4e-4,equal_nan=True)
        del outputs,output,actual,expected
        order=['A0','A1','B'];rotation=(cycle+index)%3;order=order[rotation:]+order[:rotation]
        count=max(calibrate(functions[name]) for name in order)
        iterations={name:count for name in order};gc.collect();was_enabled=gc.isenabled();gc.disable()
        try:
            timings={name:[] for name in order}
            for repeat in range(sample_count):
                rotation=repeat%3
                for name in order[rotation:]+order[:rotation]:
                    seconds=elapsed(functions[name],iterations[name]);timings[name].append(dict(iterations=iterations[name],elapsed_seconds=seconds,seconds_per_call=seconds/iterations[name]))
        finally:
            if was_enabled:gc.enable()
        median={name:statistics.median(t['seconds_per_call'] for t in timings[name]) for name in order}
        rows.append(dict(**spec,samples=timings,median_seconds=median,candidate_over_baseline=median['B']/median['A0'],control_over_baseline=median['A1']/median['A0']))
        if index % 10 == 9 or index == len(specs)-1:print('PERF_PROGRESS '+json.dumps(dict(cycle=cycle,rows=index+1,total=len(specs),key=spec['key'])),flush=True)
    return dict(cycle=cycle,platform=sys.platform,python=sys.version,numpy=np.__version__,source_hashes=source,paper_source_hashes={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in Path(fixtures).glob('*.py')},rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline-source',required=True,type=Path);parser.add_argument('--candidate-source',required=True,type=Path);parser.add_argument('--fixtures',required=True,type=Path);parser.add_argument('--cycle',type=int,default=0);parser.add_argument('--samples',type=int,default=3);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.samples<3:parser.error('--samples must be at least 3')
    result=run(args.baseline_source,args.candidate_source,args.fixtures,args.cycle,args.samples);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='completed',cycle=args.cycle,rows=len(result['rows']))),flush=True)
