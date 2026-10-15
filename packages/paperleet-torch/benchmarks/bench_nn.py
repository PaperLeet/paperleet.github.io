"""Normalization, embedding, activation and complete encoder CPU/WASM workloads.

Existing functional workloads also run against pre-module source snapshots.
New-layer cases are included only when those classes are present in the API.
Imports/fixtures are excluded; each backward call resets the relevant gradients.
"""
import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np
from paperleet_torch import __version__,torch


def workloads(dtype='float32'):
    dtype=np.dtype(dtype)
    if dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Use float32/float64')
    rng=np.random.default_rng(193)
    def tensor(shape,grad=True):
        return torch.tensor(rng.normal(size=shape).astype(dtype),requires_grad=grad)
    available={}
    for name,kind,input_grad,weight_grad,bias_grad in (
        ('layer_norm_backward','layer',True,True,True),
        ('rms_norm_backward','rms',True,True,False),
        ('layer_norm_affine_only','layer',False,True,True),
        ('rms_norm_weight_only','rms',False,True,False),
        ('layer_norm_bias_only','layer',False,False,True),
    ):
        x=tensor((16,32,128),input_grad);w=tensor((128,),weight_grad)
        b=tensor((128,),bias_grad) if kind=='layer' else None
        def run(x=x,w=w,b=b,kind=kind):
            for value in (x,w,b):
                if value is not None:value.grad=None
            output=(torch.nn.functional.layer_norm(x,(128,),w,b,1e-5) if kind=='layer'
                    else torch.nn.functional.rms_norm(x,(128,),w,1e-5))
            output.square().mean().backward()
        available[name]=run
    weights=tensor((256,32));ids=torch.tensor(rng.integers(0,256,size=(16,32)),dtype=torch.int64)
    def embedding_backward():
        weights.grad=None
        torch.nn.functional.embedding(ids,weights,padding_idx=0,scale_grad_by_freq=True).square().mean().backward()
    available['embedding_backward']=embedding_backward
    if 'GELU' not in vars(torch.nn):return available
    for approximate in ('none','tanh'):
        x=tensor((64,128));layer=torch.nn.GELU(approximate)
        def activation(x=x,layer=layer):
            x.grad=None;layer(x).square().mean().backward()
        available['gelu_'+approximate+'_backward']=activation
    dropped=tensor((64,128));dropout=torch.nn.Dropout(.1)
    def dropout_backward():
        dropped.grad=None;dropout(dropped).square().mean().backward()
    available['dropout_backward']=dropout_backward
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();N=torch.nn;options={'dtype':dtype}
            self.embed=N.Embedding(128,32,padding_idx=0,**options)
            self.norm=N.LayerNorm(32,**options);self.qkv=N.Linear(32,96,**options)
            self.rms=N.RMSNorm(32,eps=1e-5,**options)
            self.ff=N.Sequential(N.Linear(32,64,**options),N.GELU('tanh'),N.Dropout(.1),N.Linear(64,32,**options))
            self.head=N.Linear(32,5,**options)
        def forward(self,indices):
            x=self.embed(indices);batch,length=indices.shape
            q,k,v=self.qkv(self.norm(x)).reshape(batch,length,3,4,8).permute(2,0,3,1,4)
            attention=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
            x=x+attention.transpose(1,2).reshape(batch,length,32)
            return self.head((x+self.ff(self.rms(x))).mean(1))
    torch.manual_seed(193);model=Encoder();optimizer=torch.optim.SGD(model.parameters(),lr=.001,momentum=.8)
    indices=torch.tensor(rng.integers(0,128,size=(8,16)),dtype=torch.int64)
    labels=torch.tensor(np.arange(8)%5,dtype=torch.int64);loss=torch.nn.CrossEntropyLoss()
    def encoder_training_step():
        optimizer.zero_grad();loss(model(indices),labels).backward();optimizer.step()
    available['encoder_training_step']=encoder_training_step
    return available


def run_benchmarks(*,repeat=11,iterations=120,warmup=3,dtype='float32',cases=None):
    if repeat<1 or iterations<1 or warmup<0:raise ValueError('Invalid sampling configuration')
    available=workloads(dtype);names=list(available) if cases is None else list(cases)
    if not names or len(set(names))!=len(names) or any(n not in available for n in names):
        raise ValueError('Select unique available cases: '+', '.join(available))
    results={}
    for name in names:
        operation=available[name]
        for _ in range(warmup):operation()
        samples=[]
        for _ in range(repeat):
            start=time.perf_counter()
            for _ in range(iterations):operation()
            samples.append((time.perf_counter()-start)*1000/iterations)
        results[name]={'median_ms':statistics.median(samples),'samples_ms':samples}
    return dict(library_version=__version__,python=platform.python_version(),numpy=np.__version__,
                platform=platform.platform(),implementation=sys.modules[torch.Tensor.__module__].__file__,
                dtype=np.dtype(dtype).name,seed=193,repeat=repeat,iterations=iterations,warmup=warmup,results=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output');parser.add_argument('--repeat',type=int,default=11)
    parser.add_argument('--iterations',type=int,default=120)
    parser.add_argument('--dtype',choices=['float32','float64'],default='float32')
    parser.add_argument('--case',action='append',dest='cases')
    options=parser.parse_args()
    report=json.dumps(run_benchmarks(repeat=options.repeat,iterations=options.iterations,
                                    dtype=options.dtype,cases=options.cases),indent=2)+'\n'
    if options.output:
        with open(options.output,'w',encoding='utf8') as stream:stream.write(report)
    print(report)
