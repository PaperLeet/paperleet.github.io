"""Existing GroupNorm/classification API controls for joint gradient routing."""
import ast,gc,hashlib,json,statistics,time,sys,os
from pathlib import Path
import numpy as np
import bench_determinants as common

r=Path(__file__).resolve().parent

def fixture(module,kind):
    path=r/('bench_'+kind+'.py');tree=ast.parse(path.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_fixture')
    environment=dict(np=np,torch=module.torch)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),environment)
    return environment['_fixture']

def run(baseline_source,candidate_source,validate_only=False):
    old,oh=common.load('_joint_controls_old',baseline_source);new,nh=common.load('_joint_controls_new',candidate_source);rows=[]
    group=[('tiny_train',(8,16),'train'),('mlp_train',(256,128),'train'),('vision_train',(8,16,16,16),'train'),('volume_train',(4,16,4,8,8),'train'),('affine',(256,128),'affine'),('bias',(256,128),'bias'),('forward',(256,128),'forward')]
    plans=[('groupnorm',name,shape,kind) for name,shape,kind in group]
    plans += [('classification',f'{kind}_{shape[0]}',shape,kind) for shape in [(8,16),(64,32)] for kind in ['index','smooth','soft']]
    for dtype in ['float32','float64']:
        for family,name,shape,kind in plans:
            variants={};proofs={}
            for label,module in [('baseline',old),('candidate',new)]:
                operation,detail=fixture(module,family)(shape,kind,np.dtype(dtype),'joint');variants[label]=operation
                output=operation()
                if family=='groupnorm':
                    reference,inputs=fixture(module,family)(shape,kind,np.dtype(dtype),'composite');expected=reference();pairs=[(output,expected)]
                    pairs += [(a.grad,b.grad) for a,b in zip(detail,inputs) if a.grad is not None]
                    for a,b in pairs:np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=3e-5 if dtype=='float32' else 3e-12,atol=3e-5 if dtype=='float32' else 3e-12)
                    proofs[label]=dict(independent_composition_arrays=len(pairs))
                else:proofs[label]=detail
            np.testing.assert_allclose(variants['candidate']().numpy(),variants['baseline']().numpy(),rtol=3e-5 if dtype=='float32' else 3e-12,atol=3e-6 if dtype=='float32' else 3e-12)
            if validate_only:
                rows.append(dict(operation=family,case=name,kind=kind,dtype=dtype,input_shape=list(shape),reference_checks=proofs))
                continue
            gc.collect();calls=1;samples={label:[] for label in variants}
            for fn in variants.values():
                batch=1
                while True:
                    started=time.perf_counter()
                    for _ in range(batch):fn()
                    if time.perf_counter()-started>=.02:break
                    batch*=2
                calls=max(calls,batch)
            for repeat in range(5):
                for label in variants if repeat%2==0 else reversed(variants):
                    started=time.perf_counter()
                    for _ in range(calls):variants[label]()
                    samples[label].append((time.perf_counter()-started)/calls)
            medians={label:statistics.median(values) for label,values in samples.items()}
            rows.append(dict(operation=family,case=name,kind=kind,dtype=dtype,input_shape=list(shape),calls=calls,samples_s=samples,median_s=medians,candidate_over_baseline=medians['candidate']/medians['baseline'],reference_checks=proofs))
    assert len(rows)==26
    return dict(runtime=dict(python=sys.version,numpy=np.__version__,thread_environment={name:os.environ.get(name) for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}),validation_only=validate_only,source_hashes=dict(baseline=oh,candidate=nh),program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fixture_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [r/'bench_groupnorm.py',r/'bench_classification.py']},rows=rows,scope='Validation-only mode omits timing. Otherwise 26 matched existing 0.66 GroupNorm/classification controls, using frozen official fixtures and their independent composition/float64 formula checks. Actual library current/baseline APIs are timed, never those references. Both dtypes and active/frozen affine/input patterns; five alternating common >=20ms batches. Shapes/fixtures/seeds precede timing; full calls include graph creation, kernels, reduction/backward/clearing/destruction, or no-grad forward when marked. No native-Torch or GPU timing claim.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source',required=True)
    parser.add_argument('--candidate-source',required=True)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--output')
    args=parser.parse_args()
    result=run(args.baseline_source,args.candidate_source,args.validate_only)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    else:print(text,end='')
