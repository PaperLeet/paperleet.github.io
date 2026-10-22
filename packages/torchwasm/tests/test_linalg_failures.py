"""Portable factor status when NumPy cannot report floating-point flags."""
import numpy as np
import pytest
from torchwasm import torch
from torchwasm import _linalg


def singular(dtype,n,batched):
    data=np.broadcast_to(np.eye(n,dtype=dtype),(2,n,n)).copy() if batched else np.eye(n,dtype=dtype)
    if batched:data[1,-1,-1]=0
    else:data[-1,-1]=0
    return data


@pytest.mark.parametrize('dtype',['float32','float64','complex64','complex128'])
@pytest.mark.parametrize('n',[1,3,7])
@pytest.mark.parametrize('batched',[False,True])
@pytest.mark.parametrize('left',[False,True])
@pytest.mark.parametrize('empty',[False,True])
def test_singular_solve_including_empty_rhs_raises(dtype,n,batched,left,empty):
    data=singular(dtype,n,batched);width=0 if empty else 2;shape=(n,width) if left else (width,n)
    rhs=np.ones(((1,)+shape) if batched else shape,dtype=dtype)
    with pytest.raises(RuntimeError,match='singular'):torch.linalg.solve(torch.tensor(data),torch.tensor(rhs),left=left)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('n',[1,3,7])
@pytest.mark.parametrize('batched',[False,True])
def test_singular_inverse_raises(dtype,n,batched):
    with pytest.raises(RuntimeError,match='singular'):torch.linalg.inv(torch.tensor(singular(dtype,n,batched)))


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('n',[1,3,65])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('left',[False,True])
def test_triangular_zero_pivot_returns_nonfinite_values(dtype,n,upper,left):
    effective_upper=upper if left else not upper;index=n-1 if effective_upper else 0;a=np.eye(n,dtype=dtype);a[index,index]=0;shape=(n,2) if left else (2,n)
    actual=torch.linalg.solve_triangular(torch.tensor(a),torch.ones(shape,dtype=getattr(torch,dtype)),upper=upper,left=left).numpy()
    # The native API permits nonfinite results for a zero pivot; it does not
    # promise one vendor-independent distribution of NaNs versus infinities.
    assert actual.shape==shape and actual.dtype==np.dtype(dtype) and not np.isfinite(actual).all()


@pytest.mark.parametrize('operation',['solve','inv','triangular'])
def test_quiet_backend_injected_without_fpu_flags(operation):
    solve,inv=np.linalg.solve,np.linalg.inv
    def quiet(call,*args):
        try:return call(*args)
        except np.linalg.LinAlgError:
            shape=args[1].shape if len(args)==2 else args[0].shape
            return np.full(shape,np.nan,dtype=args[0].dtype)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_linalg,'_NUMPY_ERRORS_RAISE',False)
        patch.setattr(np.linalg,'solve',lambda a,b:quiet(solve,a,b))
        patch.setattr(np.linalg,'inv',lambda a:quiet(inv,a))
        a=torch.tensor(np.diag([0.,1.,1.]));b=torch.ones(3,2,dtype=torch.float64)
        if operation=='triangular':
            x=torch.linalg.solve_triangular(a,b,upper=False).numpy();assert x.shape==(3,2) and not np.isfinite(x).all()
        else:
            with pytest.raises(RuntimeError,match='singular'):
                torch.linalg.solve(a,b) if operation=='solve' else torch.linalg.inv(a)


def test_runtime_capability_probe_restores_numpy_error_mode():
    before=np.geterr()
    with np.errstate(all='raise'):
        state=np.geterr();result=_linalg._numpy_error_flags_work();assert type(result) is bool and np.geterr()==state
    assert np.geterr()==before and result==_linalg._NUMPY_ERRORS_RAISE


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_extremal_nonsingular_output_is_not_misreported_as_singular(dtype):
    magnitude=1e-30 if dtype=='float32' else 1e-300;rhs=1e30 if dtype=='float32' else 1e300
    with np.errstate(all='ignore'):
        x=torch.linalg.solve(torch.tensor(np.eye(2,dtype=dtype)*magnitude),torch.tensor(np.full((2,1),rhs,dtype=dtype)))
    assert x.shape==(2,1) and not np.isfinite(x.numpy()).all()
