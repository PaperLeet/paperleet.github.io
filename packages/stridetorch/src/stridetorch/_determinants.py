"""Stable log determinants and saved-factor real directional derivatives."""
from collections import namedtuple
import numpy as np

Slogdet = namedtuple('Slogdet', ['sign','logabsdet'])

# Older CPU NumPy and WASM LAPACK can return (0, -Inf) for a NaN pivot.
# Probe the behavior once; newer CPU builds avoid an extra input scan per call.
with np.errstate(all='ignore'):
    _SLOGDET_NAN_PROPAGATES = all(np.isnan(np.linalg.slogdet(
        np.diag(np.array([np.nan,1.],dtype=dtype)))[1]) for dtype in (np.float64,np.complex128))


def _component(value):
    data=np.asarray(value)
    return _layout.contiguous_empty(data) if not data.size else data


def _any(mask):
    # A scalar predicate already has a truth value; do not dispatch another
    # NumPy reduction for the common single-matrix path.
    return bool(mask) if mask.ndim==0 else bool(mask.any())


def _check(input):
    if not isinstance(input,_tensor.Tensor):raise TypeError('Log determinants expect a Tensor')
    if input.ndim<2 or input.shape[-2]!=input.shape[-1]:raise RuntimeError('Log determinants expect square matrices')
    if input.dtype.char not in 'fdFD':raise _errors.BrowserUnsupported('Log determinants require float32/64 or complex64/128')


def _ieee_solve(a,b,transpose=False):
    """Dense pivoted-LU fallback when LAPACK reports a singular matrix.

    Singular log derivatives are undefined. Preserve IEEE divisions rather than
    replacing zero pivots, clipping the result or treating them as zero slopes.
    """
    lu=a.copy();n=len(a);pivots=[]
    for k in range(n):
        pivot=k+int(np.argmax(np.abs(lu[k:,k])));pivots.append(pivot)
        if pivot!=k:lu[[k,pivot]]=lu[[pivot,k]]
        if lu[k,k]!=0:lu[k+1:,k]/=lu[k,k]
        lu[k+1:,k+1:]-=lu[k+1:,k,None]*lu[k,None,k+1:]
    value=b.copy()
    if not transpose:
        for k,pivot in enumerate(pivots):
            if pivot!=k:value[[k,pivot]]=value[[pivot,k]]
        for k in range(1,n):value[k]-=(lu[k,:k,None]*value[:k]).sum(axis=0)
        for k in range(n-1,-1,-1):
            value[k]-=(lu[k,k+1:,None]*value[k+1:]).sum(axis=0)
            value[k]/=lu[k,k]
    else:
        for k in range(n):
            value[k]-=(lu[:k,k,None]*value[:k]).sum(axis=0)
            value[k]/=lu[k,k]
        for k in range(n-2,-1,-1):value[k]-=(lu[k+1:,k,None]*value[k+1:]).sum(axis=0)
        for k in range(n-1,-1,-1):
            pivot=pivots[k]
            if pivot!=k:value[[k,pivot]]=value[[pivot,k]]
    return value


def _solve_values(source,rhs,transpose,*,factor_transposed=None):
    if not source.size:return np.zeros_like(rhs)
    coefficient=source.swapaxes(-1,-2) if transpose else source
    try:
        value=np.linalg.solve(coefficient,rhs)
        if _linalg._NUMPY_ERRORS_RAISE or not np.isnan(value).any():return value
        if not np.any(np.linalg.slogdet(source)[0]==0):return value
    except np.linalg.LinAlgError:pass
    value=np.empty_like(rhs);use_transpose=source.flags.c_contiguous if factor_transposed is None else factor_transposed
    for index in np.ndindex(source.shape[:-2]):
        try:
            result=np.linalg.solve(coefficient[index],rhs[index])
            if not _linalg._NUMPY_ERRORS_RAISE and np.isnan(result).any() and np.linalg.slogdet(source[index])[0]==0:raise np.linalg.LinAlgError()
        except np.linalg.LinAlgError:
            a=source[index].T if use_transpose else source[index]
            result=_ieee_solve(a,rhs[index],transpose=transpose!=use_transpose)
        value[index]=result
    return value


def _diagonal_solve(source,seed,*,factor_transposed=None):
    rhs=np.zeros_like(source);axis=np.arange(source.shape[-1]);rhs[...,axis,axis]=np.asarray(seed)[...,None]
    return _linalg._solve_layout(_solve_values(source,rhs,True,factor_transposed=factor_transposed),True)


def _log_values(source):
    with np.errstate(all='ignore'):sign,logabs=np.linalg.slogdet(source)
    if not _SLOGDET_NAN_PROPAGATES:
        # WASM LAPACK can classify a NaN pivot as singular without reporting
        # an error. Preserve NaN batches instead of silently returning -Inf.
        invalid=np.isnan(source).any(axis=(-2,-1))
        if _any(invalid):
            logabs=np.where(invalid,np.asarray(np.nan,dtype=logabs.dtype),logabs)
            if source.dtype.kind=='c':
                sign=np.where(invalid,np.asarray(complex(np.nan,np.nan),dtype=sign.dtype),sign)
    if source.dtype.kind!='c' and _any(np.isnan(logabs)):
        # Native real sgn maps a NaN pivot to zero; NumPy can report +/-1.
        sign=np.where(np.isnan(logabs),np.copysign(np.zeros_like(sign),sign),sign)
    return _component(sign),_component(logabs)


def _sign_result(value,input,logabs):
    output=_autograd.result(value,(logabs,lambda g:None),versioned=(input,))
    if output.requires_grad:
        counter=output._storage;guard=(counter,counter.version)
        output._grad_node.versions+=(guard,);logabs._grad_node.versions+=(guard,)
    return output


def _directional(primal,direction,negative=None):
    # Native 2.14.1 JVP differentiates the RHS of its saved-factor solve. Keep
    # an immutable factor snapshot and no reverse edge to the original matrix.
    source=primal._data.copy(order='K');rhs=direction._data;transposed=primal._data.flags.c_contiguous
    with np.errstate(all='ignore'):
        values=_solve_values(source,rhs,False,factor_transposed=transposed);value=np.trace(values,axis1=-2,axis2=-1)
    if negative is not None:value=np.where(negative,np.zeros((),dtype=value.dtype),value)
    def backward(g):
        if negative is not None:g=np.where(negative,np.zeros((),dtype=source.dtype),g)
        return _diagonal_solve(source,g,factor_transposed=transposed)
    return _autograd.result(_component(value),(direction,backward),versioned=())


def slogdet(input,*,out=None):
    if out is not None:raise _errors.BrowserUnsupported('Log determinant output buffers require the server')
    _check(input)
    if isinstance(input,_autograd.ForwardTensor):
        primal=slogdet(input._primal)
        tangent=_directional(input._primal,input._tangent)
        zero=_tensor.Tensor(_component(np.zeros_like(primal.sign._data)))
        return Slogdet(_autograd.ForwardTensor(primal.sign,zero),_autograd.ForwardTensor(primal.logabsdet,tangent))
    source=input._data
    sign,logabs=_log_values(source)
    if source.dtype.kind=='c':return Slogdet(_tensor.Tensor(sign),_tensor.Tensor(logabs))
    def backward(g):return _diagonal_solve(source,g)
    logabs=_autograd.result(logabs,(input,backward),versioned=(input,))
    sign=_sign_result(sign,input,logabs)
    return Slogdet(sign,logabs)


def linalg_slogdet(A,*,out=None):return slogdet(A,out=out)


def logdet(input):
    _check(input);forward=isinstance(input,_autograd.ForwardTensor)
    primal=input._primal if forward else input;source=primal._data
    sign,logabs=_log_values(source)
    if source.dtype.kind=='c':
        with np.errstate(all='ignore'):
            # Assign components separately: multiplying a NaN phase by 1j
            # would also contaminate the valid -Inf real part at singularity.
            value=np.empty(sign.shape,dtype=source.dtype)
            value.real[...]=np.where(np.isnan(sign.real)|np.isnan(sign.imag),np.nan,logabs)
            value.imag[...]=np.where(sign==0,np.nan,np.angle(sign))
        return _tensor.Tensor(_component(value))
    negative=sign<0;has_negative=_any(negative)
    value=np.where(negative,np.asarray(np.nan,dtype=source.dtype),logabs) if has_negative else logabs
    def backward(g):
        seed=np.where(negative,np.zeros((),dtype=source.dtype),g) if has_negative else g
        return _diagonal_solve(source,seed)
    output=_autograd.result(_component(value),(primal,backward),versioned=(primal,))
    return _autograd.ForwardTensor(output,_directional(primal,input._tangent,negative if has_negative else None)) if forward else output


from . import _autograd, _errors, _layout, _linalg, _tensor
