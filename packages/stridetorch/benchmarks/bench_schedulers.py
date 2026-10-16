"""Scheduler-only and complete scheduled-training costs on CPU or Pyodide.

Scheduler-only fixtures perform one optimizer step before warmup; timed calls
measure scheduler.step(), not optimizer work. No gradients are attached there.
The MLP cases include zeroing, forward, backward and SGD, with/without scheduling.
Fixtures, imports and result serialization are excluded; no CI timing thresholds.
"""
import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np


def workloads(api):
    S = api.optim.lr_scheduler
    def optimizer(groups=8, tensor_lr=False):
        return api.optim.SGD([{'params':[api.nn.Parameter(api.ones(1))],
                              'lr':api.tensor(.1/(index+1)) if tensor_lr else .1/(index+1)}
                             for index in range(groups)])
    available = {}
    for name, count, tensor, factory in (
        ('step_one_group',1,False,lambda o:S.StepLR(o,200,.9)),
        ('step_eight_groups',8,False,lambda o:S.StepLR(o,200,.9)),
        ('cosine_eight_groups',8,False,lambda o:S.CosineAnnealingLR(o,1000,1e-5)),
        ('tensor_cosine_eight_groups',8,True,lambda o:S.CosineAnnealingLR(o,1000,1e-5)),
        ('warmup_cosine_eight_groups',8,False,
         lambda o:S.SequentialLR(o,[S.LinearLR(o,.2,total_iters=5),S.CosineAnnealingLR(o,1000,1e-5)],[5])),
        ('plateau_eight_groups',8,False,lambda o:S.ReduceLROnPlateau(o,patience=5,min_lr=1e-5)),
    ):
        o = optimizer(count,tensor);s = factory(o);o.step()
        available[name] = (lambda s=s:s.step(1.)) if name.startswith('plateau') else s.step
    for scheduled in (False,True):
        api.manual_seed(412)
        model = api.nn.Sequential(api.nn.Linear(16,32),api.nn.ReLU(),api.nn.Linear(32,8))
        o = api.optim.SGD(model.parameters(),lr=.001)
        s = S.StepLR(o,200,.9) if scheduled else None
        x = api.ones(16,16)
        def train(model=model,o=o,s=s,x=x):
            o.zero_grad();model(x).square().mean().backward();o.step()
            if s is not None:s.step()
        available['scheduled_mlp' if scheduled else 'plain_mlp'] = train
    return available


def run_benchmarks(*, backend='lite', repeat=11, iterations=120, warmup=3):
    if repeat < 1 or iterations < 1 or warmup < 0:
        raise ValueError('Positive repeat/iterations and nonnegative warmup required')
    if backend == 'lite':
        from stridetorch import torch as api, __version__ as version
    elif backend == 'native':
        import torch as api
        version = api.__version__
        if getattr(api,'_browser_capabilities',None) is not None:
            raise ValueError('Native comparison must not use torch aliases')
    else:raise ValueError('Unknown backend')
    results = {}
    for name, operation in workloads(api).items():
        for _ in range(warmup):operation()
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(iterations):operation()
            samples.append((time.perf_counter()-start)*1000/iterations)
        results[name] = {'median_ms':statistics.median(samples),'samples_ms':samples}
    return dict(backend=backend,version=version,python=platform.python_version(),numpy=np.__version__,
                platform=platform.platform(),implementation=sys.modules[api.Tensor.__module__].__file__,
                repeat=repeat,iterations=iterations,warmup=warmup,dtype='float32',seed=412,results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['lite','native'],default='lite')
    parser.add_argument('--output')
    parser.add_argument('--repeat',type=int,default=11)
    parser.add_argument('--iterations',type=int,default=120)
    options = parser.parse_args()
    report = json.dumps(run_benchmarks(backend=options.backend,repeat=options.repeat,
                                     iterations=options.iterations),indent=2)+'\n'
    if options.output:
        with open(options.output,'w',encoding='utf8') as stream:stream.write(report)
    print(report)
