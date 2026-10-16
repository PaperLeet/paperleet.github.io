"""CPU checkpoint data in PyTorch's ZIP/persistent-storage representation.

The reader resolves a fixed collection of data rebuilders, never checkpoint
imports. Numerical storage is NumPy-owned; native PyTorch is not a dependency.
"""
import codecs
from collections import OrderedDict
import io
import math
import pickle
import sys

import numpy as np


class _Global:
    def __init__(self, module, name):self.module,self.name = module,name
    def __call__(self,*args):raise RuntimeError('A checkpoint global token cannot be called directly')


_STORAGE_NAMES = {'bool':'BoolStorage','int8':'CharStorage','uint8':'ByteStorage',
    'int16':'ShortStorage','int32':'IntStorage','int64':'LongStorage',
    'float16':'HalfStorage','float32':'FloatStorage','float64':'DoubleStorage',
    'complex64':'ComplexFloatStorage','complex128':'ComplexDoubleStorage'}
_DTYPES = {np.dtype(name).name:np.dtype(name) for name in
    ('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64',
     'float16','float32','float64','complex64','complex128')}
_STORAGE_DTYPES = {name:np.dtype(dtype) for dtype,name in _STORAGE_NAMES.items()}
_STORAGE_GLOBALS = {name:_Global('torch',name) for name in _STORAGE_DTYPES}
_UNTYPED = _Global('torch.storage','UntypedStorage')
_DTYPE_GLOBALS = {name:_Global('torch',name) for name in _DTYPES}
_TENSOR_V2 = _Global('torch._utils','_rebuild_tensor_v2')
_TENSOR_V3 = _Global('torch._utils','_rebuild_tensor_v3')
_PARAMETER = _Global('torch._utils','_rebuild_parameter')
_DEVICE = _Global('torch','device')


def _validate(value, seen=None, *, loaded=False):
    if seen is None:seen=set()
    if id(value) in seen:return
    seen.add(id(value))
    if isinstance(value,_autograd.ForwardTensor):
        raise _errors.BrowserUnsupported('Serializing a forward AD tensor is unsupported')
    if isinstance(value,_tensor.Tensor):
        if type(value) not in (_tensor.Tensor,_module.Parameter):
            raise _errors.BrowserUnsupported('Checkpoint tensor subclasses are unsupported')
        if loaded:
            fields={'_data','_storage','_grad_node','_requires_grad','grad'}
            valid=(set(value.__dict__)==fields and isinstance(value._data,np.ndarray)
                   and isinstance(value._storage,_tensor.StorageVersion)
                   and value._grad_node is None and value.grad is None
                   and type(value._requires_grad) is bool)
            if not valid:raise pickle.UnpicklingError('Invalid rebuilt checkpoint tensor state')
        elif set(value.__dict__)-{'_data','_storage','_grad_node','_requires_grad','grad',
                '_retain_grad','_backward_hooks','_post_accumulate_grad_hooks','_view_base','_no_grad_view'}:
            raise _errors.BrowserUnsupported('Custom tensor checkpoint attributes are unsupported')
        return
    if type(value) in (type(None),bool,int,float,complex,str,bytes,bytearray):return
    if type(value) is _device.Device:return
    if isinstance(value,np.dtype) and value.name in _DTYPES and value.metadata is None:return
    if type(value) in (list,tuple,set,frozenset):
        for item in value:_validate(item,seen,loaded=loaded)
        return
    if type(value) in (dict,OrderedDict):
        for key,item in value.items():_validate(key,seen,loaded=loaded);_validate(item,seen,loaded=loaded)
        if type(value) is OrderedDict:_validate(value.__dict__,seen,loaded=loaded)
        return
    raise _errors.BrowserUnsupported('Checkpoint data type is unsupported: '+type(value).__name__)


class _SaveStorage:
    def __init__(self,key,owner,dtype,phase,source_owner):
        self.key,self.owner,self.dtype,self.phase = key,owner,dtype,phase
        self.source_owner=source_owner
        self.nbytes = (owner.nbytes-phase)//dtype.itemsize*dtype.itemsize
        self.typed = dtype.name in _STORAGE_NAMES
        self.token = _STORAGE_GLOBALS[_STORAGE_NAMES[dtype.name]] if self.typed else _UNTYPED


def _storage_span(owner):
    if owner.flags.c_contiguous or owner.flags.f_contiguous:return owner
    # A borrowed ndarray can have padding even when its external byte buffer is
    # contiguous. Copy its bounded physical span, not its logical C-order values.
    try:backing=np.frombuffer(owner.base,dtype=np.uint8)
    except (TypeError,ValueError) as error:
        raise _errors.BrowserUnsupported('Checkpoint backing buffer is not accessible') from error
    offset=owner.__array_interface__['data'][0]-backing.__array_interface__['data'][0]
    if owner.size:
        steps=[(n-1)*s for n,s in zip(owner.shape,owner.strides)]
        begin=offset+sum(min(s,0) for s in steps)
        end=offset+sum(max(s,0) for s in steps)+owner.itemsize
    else:begin=end=offset
    if not 0<=begin<=end<=backing.nbytes:
        raise _errors.BrowserUnsupported('Checkpoint backing span exceeds its buffer')
    return backing[begin:end]


class _CheckpointPickler(pickle._Pickler):
    """Emit known native globals without changing imports or importing torch."""
    def __init__(self, file, protocol):
        super().__init__(file,protocol=protocol)
        self.storages={}

    def save(self,obj,save_persistent_id=True):
        if isinstance(obj,np.dtype):
            memo=self.memo.get(id(obj))
            if memo is not None:self.write(self.get(memo[0]));return
            self.save(_DTYPE_GLOBALS[obj.name]);self.memoize(obj);return
        if isinstance(obj,_Global):
            self.framer.commit_frame()
            memo = self.memo.get(id(obj))
            if memo is not None:self.write(self.get(memo[0]));return
            self.write(pickle.GLOBAL+obj.module.encode('ascii')+b'\n'+obj.name.encode('ascii')+b'\n')
            self.memoize(obj)
            return
        super().save(obj,save_persistent_id=save_persistent_id)

    def persistent_id(self,obj):
        if isinstance(obj,_SaveStorage):
            count = obj.nbytes//obj.dtype.itemsize if obj.typed else obj.nbytes
            return ('storage',obj.token,obj.key,'cpu',count)

    def reducer_override(self,obj):
        if isinstance(obj,_module.Parameter):
            return _PARAMETER,(obj.detach(),obj.requires_grad,OrderedDict())
        if isinstance(obj,_tensor.Tensor):
            data=obj._data;dtype=data.dtype
            if dtype.name not in _DTYPES or not dtype.isnative:
                raise _errors.BrowserUnsupported('Checkpoint tensor dtype is unsupported')
            owner=_tensor.storage_owner(data)
            strides=_layout.element_strides(data)
            if any(step<0 for step in strides):
                raise _errors.BrowserUnsupported('Negative checkpoint strides are unsupported')
            key=id(owner)
            storage=self.storages.get(key)
            raw_owner=_storage_span(owner) if storage is None else storage.owner
            delta=data.__array_interface__['data'][0]-raw_owner.__array_interface__['data'][0]
            phase=delta%dtype.itemsize
            if storage is None:
                storage=_SaveStorage(str(len(self.storages)),raw_owner,dtype,phase,owner);self.storages[key]=storage
            elif storage.dtype!=dtype or storage.phase!=phase:
                raise _errors.BrowserUnsupported('Mixed-dtype views of one checkpoint storage are unsupported')
            offset=(delta-phase)//dtype.itemsize
            _check_view(data.shape,strides,offset,storage.nbytes//dtype.itemsize)
            args=(storage,offset,data.shape,strides,obj.requires_grad,OrderedDict())
            return (_TENSOR_V2,args) if storage.typed else (_TENSOR_V3,(*args,_DTYPE_GLOBALS[dtype.name]))
        if type(obj) is _device.Device:return _DEVICE,(str(obj),)
        return NotImplemented


def _check_view(shape,strides,offset,count):
    if type(offset) is not int or offset<0:raise pickle.UnpicklingError('Invalid checkpoint storage offset')
    if type(shape) not in (tuple,list) or type(strides) not in (tuple,list) or len(shape)!=len(strides):
        raise pickle.UnpicklingError('Invalid checkpoint shape/stride metadata')
    if any(type(n) is not int or n<0 for n in (*shape,*strides)):
        raise pickle.UnpicklingError('Checkpoint dimensions and strides must be nonnegative integers')
    extent=offset if math.prod(shape)==0 else offset+1+sum((n-1)*s for n,s in zip(shape,strides))
    if extent>count:raise pickle.UnpicklingError('Checkpoint view exceeds its storage')


def save(obj, f, pickle_module=pickle, pickle_protocol=2,
         _use_new_zipfile_serialization=True, _disable_byteorder_record=False):
    import zipfile
    if pickle_module is not pickle:raise _errors.BrowserUnsupported('Custom checkpoint pickle modules are unsupported')
    if not _use_new_zipfile_serialization:raise _errors.BrowserUnsupported('Legacy checkpoint archives are unsupported')
    if type(pickle_protocol) is not int or not 2<=pickle_protocol<=pickle.HIGHEST_PROTOCOL:
        raise ValueError('Checkpoint pickle protocol must be between2 and '+str(pickle.HIGHEST_PROTOCOL))
    _validate(obj)
    buffer=io.BytesIO();encoder=_CheckpointPickler(buffer,pickle_protocol);encoder.dump(obj)
    with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        archive.writestr('archive/data.pkl',buffer.getvalue())
        archive.writestr('archive/version',b'3\n')
        if not _disable_byteorder_record:archive.writestr('archive/byteorder',sys.byteorder.encode('ascii'))
        for storage in encoder.storages.values():
            raw=storage.owner.tobytes(order='A')
            archive.writestr('archive/data/'+storage.key,raw[storage.phase:storage.phase+storage.nbytes])
    if hasattr(f,'flush'):f.flush()


class _LoadStorage:
    def __init__(self,raw,dtype,byteorder):
        self.dtype=dtype
        if dtype is None:
            if byteorder!=sys.byteorder:raise _errors.BrowserUnsupported('Non-native-endian untyped checkpoint storage is unsupported')
            self.data=np.frombuffer(raw,dtype=np.uint8).copy()
        else:
            self.data=np.frombuffer(raw,dtype=dtype.newbyteorder('<' if byteorder=='little' else '>')).astype(dtype,copy=True)


def _rebuild_tensor(storage,offset,shape,strides,requires_grad=False,backward_hooks=None,metadata=None,dtype=None):
    if not isinstance(storage,_LoadStorage):raise pickle.UnpicklingError('Invalid checkpoint storage reference')
    if metadata:raise _errors.BrowserUnsupported('Lazy conjugate/negative checkpoint metadata is unsupported')
    if backward_hooks:raise _errors.BrowserUnsupported('Serialized backward hooks are unsupported')
    dtype=storage.dtype if dtype is None else dtype
    if not isinstance(dtype,np.dtype) or dtype.name not in _DTYPES:raise pickle.UnpicklingError('Invalid checkpoint dtype')
    if type(requires_grad) is not bool:raise pickle.UnpicklingError('Invalid checkpoint gradient flag')
    if storage.data.nbytes%dtype.itemsize:raise pickle.UnpicklingError('Checkpoint storage is not aligned to its dtype')
    _check_view(shape,strides,offset,storage.data.nbytes//dtype.itemsize)
    data=np.ndarray(tuple(shape),dtype=dtype,buffer=storage.data,offset=offset*dtype.itemsize,
                    strides=tuple(s*dtype.itemsize for s in strides))
    return _tensor.Tensor(data,requires_grad=requires_grad)


def _rebuild_tensor_v3(storage,offset,shape,strides,requires_grad,backward_hooks,dtype,metadata=None):
    return _rebuild_tensor(storage,offset,shape,strides,requires_grad,backward_hooks,metadata,dtype)


def _rebuild_parameter(data,requires_grad,backward_hooks):
    if backward_hooks:raise _errors.BrowserUnsupported('Serialized parameter hooks are unsupported')
    return _module.Parameter(data,requires_grad)


def _encode_bytes(value,encoding='latin1'):
    if type(value) is not str or encoding!='latin1':raise pickle.UnpicklingError('Unsupported checkpoint byte encoding')
    return codecs.encode(value,'latin1')


class _CheckpointUnpickler(pickle._Unpickler):
    def __init__(self,file,archive,prefix,byteorder,map_location,**kwargs):
        super().__init__(file,**kwargs)
        self.archive,self.prefix,self.byteorder,self.map_location=archive,prefix,byteorder,map_location
        self.storages={}

    def get_extension(self,code):
        # EXT opcodes can use a process-wide cached callable without find_class.
        # Checkpoint data must never inherit that host-side execution capability.
        raise pickle.UnpicklingError('Checkpoint pickle extensions are unsupported')

    def load_build(self):
        state=self.stack.pop();instance=self.stack[-1]
        # Native state_dict metadata uses OrderedDict attributes. Other allowed
        # globals must not acquire state or mutate host dtype/device objects.
        if type(instance) is not OrderedDict or type(state) is not dict:
            raise pickle.UnpicklingError('Invalid rebuilt checkpoint tensor state or unsupported BUILD target')
        if any(type(key) is not str for key in state):
            raise pickle.UnpicklingError('Invalid checkpoint BUILD attribute name')
        instance.__dict__.update(state)

    dispatch=pickle._Unpickler.dispatch.copy()
    dispatch[pickle.BUILD[0]]=load_build

    def find_class(self,module,name):
        if module=='__builtin__':module='builtins'
        known={('collections','OrderedDict'):OrderedDict,('builtins','set'):set,
            ('builtins','frozenset'):frozenset,('builtins','complex'):complex,
            ('builtins','bytearray'):bytearray,('builtins','bytes'):bytes,
            ('_codecs','encode'):_encode_bytes,('torch','device'):_device.Device,
            ('torch','Size'):tuple,('torch._utils','_rebuild_tensor'):_rebuild_tensor,
            ('torch._utils','_rebuild_tensor_v2'):_rebuild_tensor,
            ('torch._utils','_rebuild_tensor_v3'):_rebuild_tensor_v3,
            ('torch._utils','_rebuild_parameter'):_rebuild_parameter}
        if (module,name) in known:return known[module,name]
        if module=='torch' and name in _STORAGE_DTYPES:return _STORAGE_DTYPES[name]
        if module=='torch' and name in _DTYPES:return _DTYPES[name]
        if module=='torch.storage' and name=='UntypedStorage':return _UNTYPED
        raise pickle.UnpicklingError('Unsupported checkpoint global: '+module+'.'+name)

    def persistent_load(self,value):
        if not isinstance(value,tuple) or len(value)!=5 or value[0] not in ('storage',b'storage'):
            raise pickle.UnpicklingError('Unsupported checkpoint persistent identifier')
        _,kind,key,location,count=value
        if type(key) is not str or not key or any(c in key for c in '/\\\x00') or key in ('.','..'):
            raise pickle.UnpicklingError('Invalid checkpoint storage key')
        if type(count) is not int or count<0:raise pickle.UnpicklingError('Invalid checkpoint storage length')
        if type(location) is not str:raise pickle.UnpicklingError('Invalid checkpoint location')
        target=(self.map_location.get(location,location) if isinstance(self.map_location,dict)
                else location if self.map_location is None else self.map_location)
        _device.check_device(target)
        dtype=None if kind is _UNTYPED else kind
        if dtype is not None and not isinstance(dtype,np.dtype):raise pickle.UnpicklingError('Invalid checkpoint storage kind')
        if key in self.storages:
            storage=self.storages[key]
            if storage.dtype!=dtype or storage.data.nbytes!=count*(dtype.itemsize if dtype is not None else 1):
                raise pickle.UnpicklingError('Conflicting checkpoint storage descriptors')
            return storage
        raw=self.archive.read(self.prefix+'data/'+key)
        if len(raw)!=count*(dtype.itemsize if dtype is not None else 1):raise pickle.UnpicklingError('Checkpoint storage length mismatch')
        storage=_LoadStorage(raw,dtype,self.byteorder);self.storages[key]=storage
        return storage


def load(f, map_location=None, pickle_module=None, *, weights_only=None, mmap=None, **pickle_load_args):
    import zipfile
    if pickle_module not in (None,pickle):raise _errors.BrowserUnsupported('Custom checkpoint pickle modules are unsupported')
    if weights_only not in (None,True):raise _errors.BrowserUnsupported('Only weights-only checkpoint data is supported')
    if mmap:raise _errors.BrowserUnsupported('Memory-mapped checkpoint loading is unsupported')
    if callable(map_location):raise _errors.BrowserUnsupported('Callable checkpoint storage mapping is unsupported')
    if map_location is not None and not isinstance(map_location,(str,dict,_device.Device)):
        raise TypeError('map_location must be a device string,device or mapping')
    if isinstance(map_location,dict) and any(not isinstance(k,str) or not isinstance(v,str) for k,v in map_location.items()):
        raise TypeError('map_location dictionary keys and values must be device strings')
    if set(pickle_load_args)-{'encoding','errors','fix_imports'}:raise TypeError('Unexpected checkpoint pickle arguments')
    try:archive=zipfile.ZipFile(f,'r')
    except zipfile.BadZipFile as error:
        raise _errors.BrowserUnsupported('Only ZIP checkpoint archives are supported') from error
    with archive:
        entries=[name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(entries)!=1:raise pickle.UnpicklingError('Expected one checkpoint pickle record')
        prefix=entries[0][:-len('data.pkl')]
        try:version=archive.read(prefix+'version').decode('ascii').strip()
        except (KeyError,UnicodeError) as error:raise pickle.UnpicklingError('Missing checkpoint archive version') from error
        if version not in ('1','2','3'):raise _errors.BrowserUnsupported('Unsupported checkpoint archive version: '+version)
        if prefix+'constants.pkl' in archive.namelist():raise _errors.BrowserUnsupported('TorchScript checkpoint loading is unsupported')
        order=archive.read(prefix+'byteorder').decode('ascii') if prefix+'byteorder' in archive.namelist() else 'little'
        if order not in ('little','big'):raise pickle.UnpicklingError('Invalid checkpoint byte order')
        options={'encoding':'utf-8',**pickle_load_args}
        value=_CheckpointUnpickler(io.BytesIO(archive.read(entries[0])),archive,prefix,order,map_location,**options).load()
    _validate(value,loaded=True)
    return value


from . import _autograd, _device, _errors, _layout, _module, _tensor
