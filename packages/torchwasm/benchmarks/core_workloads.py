"""Ordinary mixed arithmetic, power and one-backward core workloads."""
import numpy as np

def specifications():
 for dtype in ['float32','float64']:
  for size in [32,2048]:
   for mode in ['forward','backward']:
    for operation in ['arithmetic','power']:
     yield dict(key=f'{dtype}/{operation}{size}/{mode}',dtype=dtype,kind='ordinary',name=f'{operation}{size}',mode=mode,shapes=[(size,),(size,)] if operation=='arithmetic' else [(size,)])
  for name,shapes in [('linear_backward',[(32,64),(16,64),(16,)]),('attention_backward',[(1,2,32,16)]*3+[(32,32)]),('slice_backward',[(64,96)]),('indexed_backward',[(128,32),(128,)])]:
   yield dict(key=f'{dtype}/{name}',dtype=dtype,kind='ordinary',name=name,mode='backward',shapes=shapes)

def make_inputs(spec):
 rng=np.random.default_rng(295001)
 values=[]
 for i,shape in enumerate(spec['shapes']):
  if spec['name']=='indexed_backward' and i==1:
   array=rng.integers(0,64,size=shape,dtype=np.int64)
  else:array=np.asarray(rng.integers(1,9,size=shape)/8,dtype=spec['dtype'])
  array.flags.writeable=False;values.append(array)
 return values,[]

def flow(module,spec,raw,new_norm):
 api=module.torch;backward=spec['mode']=='backward';name=spec['name']
 def invoke():
  values=[api.from_numpy(array).requires_grad_(backward and array.dtype.kind=='f') for array in raw[0]]
  if name.startswith('arithmetic'):
   left,right=values
   result=(0.5-((left*right+0.375)/(right+1.25)-left/3.0))*2.0
   if not backward:return (result,)
   result=result.sum();arguments=tuple(values)
  elif name.startswith('power'):
   result=values[0].pow(3)
   if not backward:return (result,)
   result=result.sum();arguments=tuple(values)
  elif name=='linear_backward':
   result=api.nn.functional.linear(*values).square().mean();arguments=tuple(values)
  elif name=='attention_backward':
   result=api.nn.functional.scaled_dot_product_attention(*values[:3],attn_mask=values[3]).square().mean();arguments=tuple(values)
  elif name=='slice_backward':
   result=values[0][::2,8:88:2].pow(3).sum();arguments=tuple(values)
  elif name=='indexed_backward':
   result=values[0][values[1]].pow(3).sum();arguments=(values[0],)
  else:raise AssertionError(name)
  return (result,*api.autograd.grad(result,arguments))
 return invoke
