"""Native ZIP checkpoint interoperability,not another adapter as oracle."""
from collections import OrderedDict
import io
import pickle
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch as native
from stridetorch import torch as lite, UnsupportedOperation

DTYPES=['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64','complex64','complex128']


def checkpoint(lib,dtype):
 raw=np.arange(24).astype(dtype)
 if dtype=='bool':raw=np.arange(24)%2==0
 if np.dtype(dtype).kind=='c':raw=raw+1j*raw
 base=lib.tensor(raw,dtype=getattr(lib,dtype),requires_grad=np.dtype(dtype).kind=='f')
 obj=OrderedDict(base=base,again=base,step=base[3:21:2],permute=base.reshape(2,3,4).permute(1,2,0),expand=base[3:4].expand(2,3),scalar=base[4],empty=base[24:],zero=base[:0].reshape(2,0,3),singleton=base.reshape(1,24,1),parameter=lib.nn.Parameter(base,requires_grad=np.dtype(dtype).kind=='f'))
 obj._metadata={'':{'version':1},'child':{'version':2}}
 obj['primitive']={'epoch':4,'loss':.25,'complex':2+3j,'bytes':b'\x00\xffabc','set':{1,2},'shape':(2,3),'flag':True,'none':None}
 obj['dtype']=getattr(lib,dtype);obj['device']=lib.device('cpu')
 return obj


def check_restored(result,source,reader,dtype):
 assert result['base'] is result['again'];assert isinstance(result['parameter'],reader.nn.Parameter)
 assert result._metadata==source._metadata and result['primitive']==source['primitive']
 assert result['dtype']==getattr(reader,dtype) and str(result['device'])=='cpu'
 for key,value in source.items():
  if not isinstance(value,(lite.Tensor,native.Tensor)):continue
  actual=result[key];np.testing.assert_array_equal(actual.detach().numpy(),value.detach().numpy());assert actual.stride()==value.stride();assert actual.requires_grad==value.requires_grad and actual.grad is None
 with reader.no_grad():result['base'].detach()[3]=0
 assert result['step'][0].item()==0 and result['expand'][0,0].item()==0 and result['parameter'][3].item()==0


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('writer,reader',[(lite,lite),(lite,native),(native,lite)])
def test_native_weights_only_views_metadata_and_aliases(dtype,writer,reader):
 source=checkpoint(writer,dtype);file=io.BytesIO();assert writer.save(source,file) is None and not file.closed
 result=reader.load(io.BytesIO(file.getvalue()),weights_only=True,map_location='cpu');check_restored(result,source,reader,dtype)
 with zipfile.ZipFile(io.BytesIO(file.getvalue())) as archive:assert sum('/data/' in name for name in archive.namelist())==1


@pytest.mark.parametrize('protocol',[2,3,4,5])
@pytest.mark.parametrize('writer,reader',[(lite,lite),(native,lite),(lite,native)])
def test_pickle_protocols_and_native_reader_limits(protocol,writer,reader):
 source={'weight':writer.tensor([[2.,1.],[3.,4.]]),'bytes':b'abc'};file=io.BytesIO();writer.save(source,file,pickle_protocol=protocol)
 # Native's weights-only VM does not implement every protocol3-5 opcode.
 restored=reader.load(io.BytesIO(file.getvalue()),weights_only=protocol==2 if reader is native else True)
 np.testing.assert_array_equal(restored['weight'].numpy(),[[2.,1.],[3.,4.]]);assert restored['bytes']==b'abc'


@pytest.mark.parametrize('path_kind',['str','path','buffer'])
def test_file_paths_and_open_stream_ownership(tmp_path,path_kind):
 path=tmp_path/'state.pt';file=str(path) if path_kind=='str' else path if path_kind=='path' else io.BytesIO()
 lite.save({'value':lite.tensor([2.,3.])},file)
 if path_kind=='buffer':file.seek(0)
 actual=lite.load(file);np.testing.assert_array_equal(actual['value'].numpy(),[2.,3.])
 if path_kind=='buffer':assert not file.closed
 assert lite.save is lite.serialization.save and lite.load is lite.serialization.load


@pytest.mark.parametrize('writer,reader',[(lite,native),(native,lite)])
def test_model_optimizer_scheduler_resume_cross_library(writer,reader):
 def model(lib):
  layer=lib.nn.Linear(2,2).double()
  with lib.no_grad():layer.weight.copy_(lib.tensor([[.7,.2],[-.3,.5]],dtype=lib.float64));layer.bias.copy_(lib.tensor([.1,-.2],dtype=lib.float64))
  return layer
 x_values=[[.2,.4],[-.1,.7],[.5,-.3]];target_values=[[.3,.1],[.2,-.1],[.4,.2]]
 original=model(writer);optimizer=writer.optim.AdamW(original.parameters(),lr=.01,weight_decay=.03);scheduler=writer.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=.8)
 def step(lib,layer,optimizer,scheduler):
  optimizer.zero_grad();x=lib.tensor(x_values,dtype=lib.float64);target=lib.tensor(target_values,dtype=lib.float64);loss=(layer(x)-target).square().mean();loss.backward();optimizer.step();scheduler.step();return loss.item()
 for _ in range(3):step(writer,original,optimizer,scheduler)
 buffer=io.BytesIO();writer.save({'model':original.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':3},buffer)
 state=reader.load(io.BytesIO(buffer.getvalue()),weights_only=True);restored=model(reader);new_optimizer=reader.optim.AdamW(restored.parameters(),lr=.8);new_scheduler=reader.optim.lr_scheduler.StepLR(new_optimizer,2)
 restored.load_state_dict(state['model']);new_scheduler.load_state_dict(state['scheduler']);new_optimizer.load_state_dict(state['optimizer'])
 for _ in range(4):
  expected=step(writer,original,optimizer,scheduler);actual=step(reader,restored,new_optimizer,new_scheduler);np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-12)
  for a,b in zip(original.parameters(),restored.parameters()):np.testing.assert_allclose(a.detach().numpy(),b.detach().numpy(),rtol=2e-11,atol=2e-12)


def test_serialization_leaves_native_imports_untouched():
 import sys
 before={key:value for key,value in sys.modules.items() if key=='torch' or key.startswith('torch.')}
 file=io.BytesIO();lite.save({'x':lite.ones(2)},file);lite.load(io.BytesIO(file.getvalue()))
 assert all(sys.modules.get(key) is value for key,value in before.items())


@pytest.mark.parametrize('writer,reader',[(lite,lite),(lite,native),(native,lite)])
def test_external_padded_buffer_native_interop(writer,reader):
 raw=np.ndarray((3,4),dtype=np.float64,buffer=bytearray(160),offset=8,strides=(48,8));raw[:]=np.arange(12).reshape(3,4)
 x=writer.from_numpy(raw);f=io.BytesIO();writer.save({'root':x,'view':x[1:,::2]},f);result=reader.load(io.BytesIO(f.getvalue()),weights_only=True)
 np.testing.assert_array_equal(result['root'].numpy(),raw);np.testing.assert_array_equal(result['view'].numpy(),raw[1:,::2]);assert result['root'].stride()==x.stride()
 result['root'][1,0]=23.;assert result['view'][0,0].item()==23.
