"""Native and finite-difference checks of real spectral linear algebra."""
import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(3,3),(5,3),(3,5),(2,5,3),(2,3,5)])
@pytest.mark.parametrize('full',[False,True])
def test_svd_values_reconstruction_and_all_three_output_vjps(shape,dtype,full):
    rng=np.random.RandomState(32)
    data=rng.randn(*shape).astype(dtype)
    a,b=wasm.Tensor(data.copy(),requires_grad=True),torch.tensor(data,requires_grad=True)
    u,s,vh=wasm.svd(a,full);nu,ns,nvh=torch.linalg.svd(b,full_matrices=full)
    rank=s.shape[-1]
    assert u.shape==nu.shape and s.shape==ns.shape and vh.shape==nvh.shape
    assert u.dtype==s.dtype==vh.dtype==np.dtype(dtype)
    np.testing.assert_allclose(s.numpy(),ns.detach().numpy(),rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose((u.numpy()[...,:,:rank]*s.numpy()[...,None,:])@vh.numpy()[...,:rank,:],data,rtol=2e-6,atol=2e-6)
    # Eigenvector signs can differ across LAPACK implementations. Align only
    # the constant upstream weights, not either implementation's derivative.
    signs=np.sign((u.numpy()[...,:,:rank]*nu.detach().numpy()[...,:,:rank]).sum(axis=-2))
    gu=rng.randn(*u.shape).astype(dtype);gs=rng.randn(*s.shape).astype(dtype);gh=rng.randn(*vh.shape).astype(dtype)
    native_gu,native_gh=gu.copy(),gh.copy()
    native_gu[...,:,:rank]*=signs[...,None,:];native_gh[...,:rank,:]*=signs[...,:,None]
    ((u*wasm.Tensor(gu)).sum()+(s*wasm.Tensor(gs)).sum()+(vh*wasm.Tensor(gh)).sum()).backward()
    ((nu*torch.tensor(native_gu)).sum()+(ns*torch.tensor(gs)).sum()+(nvh*torch.tensor(native_gh)).sum()).backward()
    np.testing.assert_allclose(a.grad.numpy(),b.grad.numpy(),rtol=8e-5,atol=8e-5)
    np.testing.assert_array_equal(a.numpy(),data)


@pytest.mark.parametrize('shape',[(2,3),(3,2)])
def test_polar_gradient_also_matches_independent_central_finite_differences(shape):
    rng=np.random.RandomState(13);data=rng.randn(*shape);weights=rng.randn(*shape)
    a=wasm.Tensor(data.copy(),requires_grad=True)
    u,s,vh=wasm.svd(a,False)
    ((u@vh)*wasm.Tensor(weights)).sum().backward()
    finite=np.empty_like(data)
    def objective(value):
        left,_,right=np.linalg.svd(value,full_matrices=False)
        return ((left@right)*weights).sum()
    for index in np.ndindex(data.shape):
        lo,hi=data.copy(),data.copy();lo[index]-=1e-6;hi[index]+=1e-6
        finite[index]=(objective(hi)-objective(lo))/2e-6
    np.testing.assert_allclose(a.grad.numpy(),finite,rtol=2e-6,atol=2e-6)


@pytest.mark.parametrize('shape',[(0,3),(3,0),(2,0,3),(0,3,2)])
def test_svd_empty_axes_and_batches_preserve_shapes_and_zero_gradients(shape):
    for full in (True,False):
        a,b=wasm.Tensor(np.empty(shape),requires_grad=True),torch.empty(shape,dtype=torch.float64,requires_grad=True)
        u,s,vh=wasm.svd(a,full);nu,ns,nvh=torch.linalg.svd(b,full_matrices=full)
        assert (u.shape,s.shape,vh.shape)==(nu.shape,ns.shape,nvh.shape)
        (u.sum()+s.sum()+vh.sum()).backward();(nu.sum()+ns.sum()+nvh.sum()).backward()
        np.testing.assert_array_equal(a.grad.numpy(),b.grad.numpy())


def test_svd_strides_rank_deficiency_and_repeated_values_retain_forward_products():
    for data in (np.diag([3.,3.,0.]),np.zeros((4,3)),np.diag([1e-20,2e-20]),np.arange(30.).reshape(5,6)[:,::2]):
        actual=wasm.svd(wasm.Tensor(data),False)
        expected=torch.linalg.svd(torch.tensor(data),full_matrices=False)
        np.testing.assert_allclose(actual.S.numpy(),expected.S.numpy(),atol=1e-12)
        np.testing.assert_allclose((actual.U.numpy()*actual.S.numpy())@actual.Vh.numpy(),data,atol=1e-12)
    for data in (np.eye(3),np.diag([1.,0.]),np.diag([1.,1.+1e-16])):
        a=wasm.Tensor(data,requires_grad=True)
        u,_,_=wasm.svd(a,False)
        with pytest.raises(wasm.BrowserUnsupported,match='SVD vector gradients'): u.sum().backward()
        assert a.grad is None


def test_svd_singular_value_gradients_and_saved_outputs_are_guarded():
    data=np.random.RandomState(91).randn(4,3)
    a,b=wasm.Tensor(data.copy(),requires_grad=True),torch.tensor(data,requires_grad=True)
    wasm.svd(a,False).S.square().sum().backward()
    torch.linalg.svdvals(b).square().sum().backward()
    np.testing.assert_allclose(a.grad.numpy(),b.grad.numpy(),atol=1e-12)
    for changed in ('U','S','Vh'):
        x=wasm.Tensor(data.copy(),requires_grad=True);decomposition=wasm.svd(x,False)
        loss=decomposition.U.sum()+decomposition.S.sum()+decomposition.Vh.sum()
        getattr(decomposition,changed).detach()[...]=0.
        with pytest.raises(wasm.BrowserUnsupported,match='saved for backward'): loss.backward()
        assert x.grad is None


@pytest.mark.parametrize('scale',[1e-200,1e200])
def test_extreme_spectral_vector_scales_fall_back_instead_of_returning_nan_gradients(scale):
    a=wasm.Tensor(np.diag([scale,2*scale]),requires_grad=True)
    u,_,_=wasm.svd(a,False)
    with pytest.raises(wasm.BrowserUnsupported): u.sum().backward()
    assert a.grad is None


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('triangle',['L','U'])
def test_eigh_is_an_independent_triangle_aware_value_oracle(dtype,triangle):
    raw=np.random.RandomState(44).randn(2,4,4).astype(dtype)
    actual=wasm.eigh(wasm.Tensor(raw),triangle)
    expected=torch.linalg.eigh(torch.tensor(raw),UPLO=triangle)
    np.testing.assert_allclose(actual.eigenvalues.numpy(),expected.eigenvalues.numpy(),rtol=2e-6,atol=2e-6)
    triangular=np.tril(raw) if triangle=='L' else np.triu(raw)
    symmetric=triangular+np.swapaxes(triangular,-1,-2)
    idx=np.arange(4);symmetric[...,idx,idx]=raw[...,idx,idx]
    q=actual.eigenvectors.numpy()
    np.testing.assert_allclose((q*actual.eigenvalues.numpy()[...,None,:])@np.swapaxes(q,-1,-2),symmetric,atol=2e-6)


@pytest.mark.parametrize('shape',[(4,3),(3,4),(2,4,3)])
@pytest.mark.parametrize('mode',['reduced','complete'])
def test_qr_basis_reconstructs_inputs_without_assuming_phase_identity(shape,mode):
    data=np.random.RandomState(39).randn(*shape)
    actual=wasm.qr(wasm.Tensor(data),mode)
    expected=torch.linalg.qr(torch.tensor(data),mode=mode)
    assert actual.Q.shape==expected.Q.shape and actual.R.shape==expected.R.shape
    q,r=actual.Q.numpy(),actual.R.numpy()
    np.testing.assert_allclose(q@r,data,atol=1e-12)
    np.testing.assert_allclose(np.swapaxes(q,-1,-2)@q,np.broadcast_to(np.eye(q.shape[-1]),q.shape[:-2]+(q.shape[-1],q.shape[-1])),atol=1e-12)


def test_unsupported_spectral_modes_never_silently_drop_gradients():
    a=wasm.Tensor(np.eye(3),requires_grad=True)
    for function in (wasm.qr,):
        with pytest.raises(wasm.BrowserUnsupported): function(a)
        with wasm.no_grad():
            assert all(not tensor.requires_grad for tensor in function(a))
    for options in ({'driver':'gesvd'},{'out':(a,a,a)}):
        with pytest.raises(wasm.BrowserUnsupported): wasm.svd(a,**options)
    with pytest.raises(wasm.BrowserUnsupported): wasm.svd(wasm.Tensor(np.eye(3,dtype=np.complex128)))
    with pytest.raises(RuntimeError): wasm.svd(wasm.Tensor([1.,2.]))
