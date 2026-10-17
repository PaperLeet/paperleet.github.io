"""Actual 0.52 core vs candidate core, plus standard Transformer layer overhead."""
import importlib.util
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np


def load(name,path):
    path=Path(path)
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in path.iterdir() if f.suffix=='.py' or f.name in ['LICENSE','NOTICE']}
    return module,hashes


def run(baseline_source,candidate_source,samples=5,calls=4):
    old,oh=load('_transformer_baseline',baseline_source);new,nh=load('_transformer_candidate',candidate_source)
    rng=np.random.default_rng(5314);rows=[]
    def manual(t,embed,heads,hidden,pre,dtype):
        class Encoder(t.nn.Module):
            def __init__(self):
                super().__init__();opts=dict(dtype=dtype)
                self.self_attn=t.nn.MultiheadAttention(embed,heads,batch_first=True,**opts)
                self.linear1=t.nn.Linear(embed,hidden,**opts);self.dropout=t.nn.Dropout(0.)
                self.linear2=t.nn.Linear(hidden,embed,**opts)
                self.norm1=t.nn.LayerNorm(embed,**opts);self.norm2=t.nn.LayerNorm(embed,**opts)
                self.dropout1=t.nn.Dropout(0.);self.dropout2=t.nn.Dropout(0.);self.activation=t.nn.GELU()
            def forward(self,x):
                def sa(q):return self.dropout1(self.self_attn(q,q,q,need_weights=False)[0])
                def ff(q):return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(q)))))
                if pre:
                    y=x+sa(self.norm1(x));return y+ff(self.norm2(y))
                y=self.norm1(x+sa(x));return self.norm2(y+ff(y))
        return Encoder()
    for dtype in ['float32','float64']:
        for pre in [False,True]:
            for label,batch,length,embed,heads in [('small',2,8,8,2),('encoder',2,32,64,4),('wide',1,32,128,4)]:
                hidden=2*embed;models={};variants={'baseline':old.torch,'candidate_control':new.torch,'candidate_layer':new.torch}
                old.torch.manual_seed(5314);new.torch.manual_seed(5314)
                models['baseline']=manual(old.torch,embed,heads,hidden,pre,getattr(old.torch,dtype))
                models['candidate_control']=manual(new.torch,embed,heads,hidden,pre,getattr(new.torch,dtype))
                models['candidate_layer']=new.torch.nn.TransformerEncoderLayer(embed,heads,hidden,dropout=0.,activation='gelu',norm_first=pre,batch_first=True,dtype=getattr(new.torch,dtype))
                state={name:p.numpy().copy() for name,p in models['baseline'].named_parameters()}
                for label2,m in models.items():
                    t=variants[label2]
                    with t.no_grad():
                        for name,p in m.named_parameters():p.copy_(t.tensor(state[name]))
                values=rng.normal(0,.3,size=(batch,length,embed)).astype(dtype)
                direction=rng.normal(0,.2,size=values.shape).astype(dtype);seed=rng.normal(size=values.shape).astype(dtype)
                target=rng.normal(0,.3,size=values.shape).astype(dtype)
                qualified=None;mixed_qualified=None
                for label2,m in models.items():
                    t=variants[label2];x=t.tensor(values,requires_grad=True);y=m(x)
                    g=t.autograd.grad((y*t.tensor(seed)).sum(),[x,*m.parameters()])
                    p,j=t.func.jvp(m,(x,),(t.tensor(direction),))
                    evidence=[v.numpy().copy() for v in (y,*g,p,j)]
                    if qualified is None:qualified=evidence
                    else:
                        for actual,want in zip(evidence,qualified):np.testing.assert_allclose(actual,want,rtol=8e-5 if dtype=='float32' else 3e-10,atol=8e-6 if dtype=='float32' else 3e-11)
                    d=t.tensor(direction,requires_grad=True)
                    p,j=t.func.jvp(m,(x,),(d,))
                    mixed=t.autograd.grad((p*t.tensor(seed)+j*t.tensor(target)).sum(),[x,*m.parameters(),d])
                    evidence=[v.numpy().copy() for v in mixed]
                    if mixed_qualified is None:mixed_qualified=evidence
                    else:
                        for actual,want in zip(evidence,mixed_qualified):
                            if dtype=='float64':np.testing.assert_allclose(actual,want,rtol=3e-10,atol=3e-11)
                            else:assert np.max(abs(actual-want),initial=0)<=5e-5*max(1.,np.max(abs(want),initial=0))
                def restore(name):
                    t=variants[name]
                    with t.no_grad():
                        for n,p in models[name].named_parameters():p.copy_(t.tensor(state[n]))
                def training(name,verify=False):
                    t=variants[name];m=models[name];o=t.optim.SGD(m.parameters(),lr=.003,momentum=.7)
                    for _ in range(4):
                        o.zero_grad();(m(t.tensor(values))-t.tensor(target)).square().mean().backward();o.step()
                    if verify:return [p.numpy().copy() for p in m.parameters()]
                trained=None
                for name in models:
                    restore(name);result=training(name,True)
                    if trained is None:trained=result
                    else:
                        for actual,want in zip(result,trained):np.testing.assert_allclose(actual,want,rtol=8e-5 if dtype=='float32' else 3e-10,atol=8e-6 if dtype=='float32' else 3e-11)
                    restore(name)
                def invoke(name,mode):
                    t=variants[name];m=models[name]
                    if mode=='training4':return training(name)
                    x=t.tensor(values,requires_grad=mode=='reverse')
                    if mode=='forward':
                        with t.no_grad():m(x)
                    elif mode=='reverse':t.autograd.grad((m(x)*t.tensor(seed)).sum(),[x,*m.parameters()])
                    elif mode=='mixed':
                        x.requires_grad_();d=t.tensor(direction,requires_grad=True)
                        p,j=t.func.jvp(m,(x,),(d,))
                        t.autograd.grad((p*t.tensor(seed)+j*t.tensor(target)).sum(),[x,*m.parameters(),d])
                    else:t.func.jvp(m,(x,),(t.tensor(direction),))
                for mode in ['forward','reverse','jvp','mixed','training4']:
                    for name in models:invoke(name,mode)
                    times={name:[] for name in models};repeat=1 if mode=='training4' else calls
                    for sample in range(samples):
                        order=list(models) if sample%2==0 else list(reversed(models))
                        for name in order:
                            if mode=='training4':restore(name)
                            started=time.perf_counter()
                            for _ in range(repeat):invoke(name,mode)
                            times[name].append((time.perf_counter()-started)/repeat)
                    medians={name:statistics.median(v) for name,v in times.items()}
                    rows.append(dict(case=label,dtype=dtype,pre_norm=pre,mode=mode,samples_s=times,median_s=medians,
                        candidate_core_over_baseline=medians['candidate_control']/medians['baseline'],
                        candidate_layer_over_baseline=medians['candidate_layer']/medians['baseline']))
    return dict(scope='Actual 0.52 core vs candidate core through identical manual encoder equations; standard candidate layer measured separately. LayerNorm FP32 precision changes are included. No native Torch/GPU timing claim.',
                source_hashes=dict(baseline=oh,candidate=nh),samples=samples,calls=calls,rows=rows)


if __name__=='__main__':print(json.dumps(run(sys.argv[1],sys.argv[2]),indent=2))
