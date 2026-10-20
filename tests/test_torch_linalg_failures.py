"""Native finite singular errors and triangular IEEE values across layouts."""
import numpy as np
import pytest
import torch as native
from torchwasm import torch as lite


@pytest.mark.parametrize('dtype',['float32','float64','complex64','complex128'])
@pytest.mark.parametrize('n',[1,3,7])
@pytest.mark.parametrize('batched',[False,True])
@pytest.mark.parametrize('left',[False,True])
@pytest.mark.parametrize('empty',[False,True])
def test_finite_singular_solve_errors_match_native(dtype,n,batched,left,empty):
    data=np.broadcast_to(np.eye(n,dtype=dtype),(2,n,n)).copy() if batched else np.eye(n,dtype=dtype)
    if batched:data[1,-1,-1]=0
    else:data[-1,-1]=0
    width=0 if empty else 2;shape=(n,width) if left else (width,n);rhs=np.ones(((1,)+shape) if batched else shape,dtype=dtype)
    for lib in (lite,native):
        with pytest.raises(RuntimeError,match='singular'):lib.linalg.solve(lib.tensor(data),lib.tensor(rhs),left=left)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('n',[1,3,7])
@pytest.mark.parametrize('batched',[False,True])
def test_finite_singular_inverse_errors_match_native(dtype,n,batched):
    data=np.broadcast_to(np.eye(n,dtype=dtype),(2,n,n)).copy() if batched else np.eye(n,dtype=dtype)
    if batched:data[1,-1,-1]=0
    else:data[-1,-1]=0
    for lib in (lite,native):
        with pytest.raises(RuntimeError,match='singular'):lib.linalg.inv(lib.tensor(data))


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('n',[1,3,65])
@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('left',[False,True])
def test_triangular_zero_pivot_nonfinite_status_matches_native(dtype,n,upper,left):
    data=np.eye(n,dtype=dtype);effective_upper=upper if left else not upper;data[-1 if effective_upper else 0,-1 if effective_upper else 0]=0;shape=(n,2) if left else (2,n)
    actual=lite.linalg.solve_triangular(lite.tensor(data),lite.ones(shape,dtype=getattr(lite,dtype)),upper=upper,left=left)
    expected=native.linalg.solve_triangular(native.tensor(data),native.ones(shape,dtype=getattr(native,dtype)),upper=upper,left=left)
    assert actual.shape==expected.shape and actual.dtype==np.dtype(dtype)
    assert not np.isfinite(actual.numpy()).all() and not np.isfinite(expected.numpy()).all()
