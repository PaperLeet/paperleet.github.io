"""Native-reference pooling geometry, IEEE selection, dual tapes and CNN updates."""
import numpy as np
import pytest
import torch
from stridetorch import torch as lite


@pytest.fixture(params=['scan', 'window', 'block', 'auto'], autouse=True)
def pooling_algorithm(request, monkeypatch):
    from stridetorch import _pooling
    monkeypatch.setattr(_pooling, '_ALGORITHM', request.param)


def array(value):return value.detach().numpy()


def close(a, b, dtype=np.float64):
    np.testing.assert_allclose(array(a), array(b), rtol=5e-5 if dtype==np.float32 else 2e-11,
                               atol=2e-5 if dtype==np.float32 else 2e-12, equal_nan=True)


CASES = [
    ((2, 3, 5, 7), dict(kernel_size=2)),
    ((2, 3, 5, 7), dict(kernel_size=3, stride=1, padding=1)),
    ((2, 3, 5, 7), dict(kernel_size=(2, 3), stride=(2, 2), padding=(1, 1), ceil_mode=True)),
    ((2, 3, 7, 9), dict(kernel_size=3, dilation=2, padding=1, stride=1)),
    ((3, 5, 7), dict(kernel_size=(2,), stride=[], ceil_mode=True)),
    ((0, 3, 5, 7), dict(kernel_size=2)),
    ((1, 2, 4, 5), dict(kernel_size=2, dilation=2, stride=1, padding=1, ceil_mode=True)),
]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('layout', ['contiguous', 'fortran', 'strided'])
def test_values_indices_seeded_gradients_and_retained_graph(dtype, case, layout):
    shape, kw = case;rng = np.random.default_rng(3802)
    a = rng.integers(-3, 4, size=shape).astype(dtype)
    if layout=='fortran':a=np.asfortranarray(a)
    if layout=='strided':a=np.repeat(a, 2, axis=-1)[..., ::2]
    records=[];seed=None
    for api in (lite, torch):
        x=api.from_numpy(a).requires_grad_();y,i=api.nn.functional.max_pool2d(x, **kw, return_indices=True)
        if seed is None:seed=rng.normal(size=y.shape).astype(dtype)
        y.backward(api.tensor(seed), retain_graph=True);y.backward(api.tensor(seed))
        records.append((y,i,x.grad))
    close(records[0][0],records[1][0],dtype)
    np.testing.assert_array_equal(array(records[0][1]),array(records[1][1]))
    np.testing.assert_array_equal(array(records[0][2]),array(records[1][2]))
    assert records[0][0].dtype==np.dtype(dtype) and records[0][1].dtype==lite.int64


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8])
@pytest.mark.parametrize('padding', [0, 1])
@pytest.mark.parametrize('stride', [1, 2])
def test_ieee_minimum_ties_nan_payloads_and_index_dtype(dtype, padding, stride):
    if np.dtype(dtype).kind=='f':
        cases=[[-0.,0.,0.,-0.],[-np.inf]*4,[np.nan,1.,np.nan,0.],[np.inf,np.nan,np.inf,np.nan]]
    else:cases=[[np.iinfo(dtype).min]*4,[2,2,2,2]]
    for values in cases:
        a=np.asarray(values,dtype=dtype).reshape(1,1,2,2);records=[]
        for api in (lite,torch):records.append(api.nn.functional.max_pool2d(api.from_numpy(a),2,stride,padding,return_indices=True))
        np.testing.assert_array_equal(array(records[0][1]),array(records[1][1]))
        np.testing.assert_allclose(array(records[0][0]),array(records[1][0]),rtol=0,atol=0,equal_nan=True)
        if np.dtype(dtype).kind=='f':np.testing.assert_array_equal(np.signbit(array(records[0][0])),np.signbit(array(records[1][0])))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('kw', [dict(kernel_size=2),dict(kernel_size=3,stride=1,padding=1)])
@pytest.mark.parametrize('seed_values', [[-0.,-0.,-0.,-0.],[np.nan,np.inf,-np.inf,1.]])
def test_nonfinite_and_signed_zero_seed_accumulation(dtype, kw, seed_values):
    a=np.arange(16,dtype=dtype).reshape(1,1,4,4);records=[]
    for api in (lite,torch):
        x=api.from_numpy(a).requires_grad_();y=api.nn.functional.max_pool2d(x,**kw)
        seed=np.resize(np.asarray(seed_values,dtype=dtype),y.shape)
        with np.errstate(invalid='ignore'):y.backward(api.tensor(seed))
        records.append(x.grad)
    np.testing.assert_allclose(array(records[0]),array(records[1]),rtol=0,atol=0,equal_nan=True)
    np.testing.assert_array_equal(np.signbit(array(records[0])),np.signbit(array(records[1])))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('case', CASES[:5])
@pytest.mark.parametrize('recording', [True, False])
def test_jvp_primal_tangent_indices_and_reverse_over_forward(dtype,case,recording):
    shape,kw=case;rng=np.random.default_rng(3803);a=rng.normal(size=shape).astype(dtype);v=rng.normal(size=shape).astype(dtype);records=[]
    for api in (lite,torch):
        x=api.tensor(a,requires_grad=True);direction=api.tensor(v,requires_grad=True)
        with api.set_grad_enabled(recording):
            (p,i),(t,ti)=api.func.jvp(lambda z:api.nn.functional.max_pool2d(z,**kw,return_indices=True),(x,),(direction,))
        if recording:t.square().sum().backward();assert x.grad is None
        else:assert not p.requires_grad and not t.requires_grad
        records.append((p,i,t,ti,direction.grad))
    for position in (0,2):close(records[0][position],records[1][position],dtype)
    for position in (1,3):np.testing.assert_array_equal(array(records[0][position]),array(records[1][position]))
    if recording:close(records[0][4],records[1][4],dtype)


@pytest.mark.parametrize('target', ['input','indices','output','tangent','primal_output','tangent_output'])
def test_native_saved_version_boundaries(target):
    records=[]
    for api in (lite,torch):
        x=api.arange(9,dtype=api.float64).reshape(1,1,3,3).requires_grad_()
        if target in ('input','indices','output'):
            p,i=api.nn.functional.max_pool2d(x,2,1,return_indices=True);t=p;v=x
        else:
            v=api.ones_like(x,requires_grad=True)
            (p,i),(t,ti)=api.func.jvp(lambda z:api.nn.functional.max_pool2d(z,2,1,return_indices=True),(x,),(v,))
        with api.no_grad():{'input':x,'indices':i,'output':p,'tangent':v,'primal_output':p,'tangent_output':t}[target].detach().add_(1)
        try:t.sum().backward();records.append('success')
        except RuntimeError:records.append('RuntimeError')
    assert records[0]==records[1]


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
def test_complete_cnn_two_real_sgd_updates(dtype):
    rng=np.random.default_rng(3804);input_data=rng.normal(size=(2,2,6,8)).astype(dtype);target=rng.normal(size=(2,4)).astype(dtype)
    initial=None;records=[]
    for api in (lite,torch):
        model=api.nn.Sequential(api.nn.Conv2d(2,3,3,padding=1),api.nn.ReLU(),api.nn.MaxPool2d(2),api.nn.Flatten(),api.nn.Linear(36,4))
        model=model.to(dtype=api.float32 if dtype==np.float32 else api.float64)
        parameters=list(model.parameters())
        if initial is None:initial=[rng.normal(size=p.shape).astype(dtype)*.1 for p in parameters]
        with api.no_grad():
            for p,a in zip(parameters,initial):p.copy_(api.tensor(a))
        optimizer=api.optim.SGD(parameters,lr=.02,momentum=.7);steps=[]
        for _ in range(2):
            x=api.tensor(input_data,requires_grad=True);optimizer.zero_grad(set_to_none=True)
            y=model(x);loss=api.nn.functional.mse_loss(y,api.tensor(target));loss.backward()
            steps.append([array(y).copy(),array(loss).copy(),array(x.grad).copy(),*[array(p.grad).copy() for p in parameters]])
            optimizer.step()
        records.append((steps,[array(p).copy() for p in parameters]))
    for left,right in zip(records[0][0],records[1][0]):
        for a,b in zip(left,right):np.testing.assert_allclose(a,b,rtol=8e-5 if dtype==np.float32 else 2e-10,atol=2e-6 if dtype==np.float32 else 2e-12)
    for a,b in zip(records[0][1],records[1][1]):np.testing.assert_allclose(a,b,rtol=5e-5 if dtype==np.float32 else 2e-11,atol=2e-7 if dtype==np.float32 else 2e-12)


@pytest.mark.parametrize('kw', [dict(kernel_size=0),dict(kernel_size=True),dict(kernel_size=(2,2,2)),dict(kernel_size=2,stride=0),dict(kernel_size=2,padding=2),dict(kernel_size=2,ceil_mode=1)])
def test_native_argument_failure_types(kw):
    errors=[]
    for api in (lite,torch):
        with pytest.raises(Exception) as observed:api.nn.functional.max_pool2d(api.ones(1,1,4,4),**kw)
        errors.append(type(observed.value))
    assert errors[0]==errors[1]


def test_native_empty_dilation_window_inference():
    records=[api.nn.functional.max_pool2d(api.ones(1,1,1,1),2,padding=1,dilation=2,return_indices=True) for api in (lite,torch)]
    for a,b in zip(*records):np.testing.assert_array_equal(array(a),array(b))
