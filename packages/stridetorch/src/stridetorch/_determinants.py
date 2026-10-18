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


def _ieee_solve(a,b,transpose=False,*,regularize=False):
    """Dense pivoted-LU fallback when LAPACK reports a singular matrix.

    Singular log derivatives are undefined. Preserve IEEE divisions rather than
    replacing zero pivots, clipping the result or treating them as zero slopes.
    Only determinant JVP explicitly opts into the native zero-pivot epsilon rule.
    """
    lu=a.copy();n=len(a);pivots=[]
    for k in range(n):
        pivot=k+int(np.argmax(np.abs(lu[k:,k])));pivots.append(pivot)
        if pivot!=k:lu[[k,pivot]]=lu[[pivot,k]]
        if lu[k,k]!=0:lu[k+1:,k]/=lu[k,k]
        lu[k+1:,k+1:]-=lu[k+1:,k,None]*lu[k,None,k+1:]
    if regularize:
        axis=np.arange(n);diagonal=lu[axis,axis]
        lu[axis,axis]=np.where(diagonal==0,np.finfo(lu.dtype).eps,diagonal)
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


def _solve_values(source,rhs,transpose,*,factor_transposed=None,regularize=False):
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
            result=_ieee_solve(a,rhs[index],transpose=transpose!=use_transpose,regularize=regularize)
        value[index]=result
    return value


def _diagonal_solve(source,seed,*,factor_transposed=None,regularize=False):
    rhs=np.zeros_like(source);axis=np.arange(source.shape[-1]);rhs[...,axis,axis]=np.asarray(seed)[...,None]
    if regularize and source.shape[-1]==1:
        # Native float32 triangular solves take the reciprocal in float32;
        # NumPy's promoted solve otherwise changes subnormal IEEE results.
        coefficient=np.where(source==0,np.finfo(source.dtype).eps,source)
        return _linalg._solve_layout(np.reciprocal(coefficient)*rhs,True)
    return _linalg._solve_layout(_solve_values(source,rhs,True,factor_transposed=factor_transposed,regularize=regularize),True)


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
    if _state._hvp_active:
        solved=_linalg.solve(primal,direction)
        value=solved.diagonal(dim1=-2,dim2=-1).sum(-1)
        return _ops.where(_tensor.Tensor(negative),0.,value) if negative is not None else value
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


def _cofactor_direction(source, direction):
    """Adjugate and its directional derivative without a dense Hessian.

    Products excluding one/two singular values remain defined at zero pivots.
    The rotated off-diagonal derivative simplifies to -prod(except i,j)*M[j,i],
    avoiding divisions by squared singular-value gaps. This polynomial
    derivative stays defined even when an SVD basis itself is not unique.
    """
    u,s,vh=np.linalg.svd(source,full_matrices=False)
    n=s.shape[-1];ones=np.ones(s.shape[:-1]+(1,),dtype=s.dtype)
    prefix=np.concatenate((ones,np.cumprod(s,axis=-1)),axis=-1)
    suffix=np.concatenate((np.flip(np.cumprod(np.flip(s,axis=-1),axis=-1),axis=-1),ones),axis=-1)
    cofactors=prefix[...,:-1]*suffix[...,1:]
    pairs=np.zeros(s.shape+(n,),dtype=s.dtype);axis=np.arange(n)
    for i in range(n):
        mask=axis!=i;other=s[...,mask]
        before=np.concatenate((ones,np.cumprod(other,axis=-1)),axis=-1)
        after=np.concatenate((np.flip(np.cumprod(np.flip(other,axis=-1),axis=-1),axis=-1),ones),axis=-1)
        pairs[...,i,mask]=before[...,:-1]*after[...,1:]
    rotated=np.swapaxes(u,-1,-2)@direction@np.swapaxes(vh,-1,-2)
    core=-pairs*np.swapaxes(rotated,-1,-2)
    core[...,axis,axis]=np.sum(pairs*np.diagonal(rotated,axis1=-2,axis2=-1)[...,None,:],axis=-1)
    orientation=np.linalg.det(u)*np.linalg.det(vh)
    cofactor=orientation[...,None,None]*((u*cofactors[...,None,:])@vh)
    curvature=orientation[...,None,None]*(u@core@vh)
    return cofactor,curvature


def _det_hvp_factors(primal, direction, determinant, singular):
    n=primal.shape[-1]
    source=primal._data.reshape((-1,n,n));vector=direction._data.reshape((-1,n,n))
    mask=singular.reshape(-1);value=determinant.reshape(-1)
    cofactors=np.empty_like(source);curvature=np.empty_like(source)
    with np.errstate(all='ignore'):
        if np.any(mask):cofactors[mask],curvature[mask]=_cofactor_direction(source[mask],vector[mask])
        if np.any(~mask):
            a=source[~mask];e=vector[~mask]
            inverse=np.linalg.solve(a,np.broadcast_to(np.eye(n,dtype=a.dtype),a.shape))
            transposed=np.swapaxes(inverse,-1,-2)
            c=value[~mask,None,None]*transposed
            cofactors[~mask]=c
            trace=np.trace(inverse@e,axis1=-2,axis2=-1)
            curvature[~mask]=trace[...,None,None]*c-(transposed@np.swapaxes(e,-1,-2))@c
        cofactors=cofactors.reshape(primal.shape);curvature=curvature.reshape(primal.shape)
        tangent_value=np.sum(cofactors*direction._data,axis=(-2,-1))
    def scale(g,value):return np.asarray(g)[...,None,None]*value
    output=_autograd.result(determinant,(primal,lambda g:scale(g,cofactors)),versioned=(primal,))
    tangent=_autograd.result(_component(tangent_value),(primal,lambda g:scale(g,curvature)),
        (direction,lambda g:scale(g,cofactors)),versioned=(primal,direction))
    return _autograd.ForwardTensor(output,tangent)


def det_jvp(input):
    primal=input._primal;direction=input._tangent;output=_linalg.det(primal)
    if _state._hvp_active:
        size=primal.shape[-1]
        if not primal.numel():
            # Native det's empty backward is a constant zero, including on its
            # higher-order tape. Keep that independence in strict HVP mode.
            return _autograd.ForwardTensor(output.detach(),_tensor.Tensor(np.zeros_like(output._data)))
        if size>=2:
            singular=np.abs(output._data)<100*np.finfo(primal.dtype).eps
            if np.any(singular):return _det_hvp_factors(primal,direction,output._data,singular)
        if size==1:tangent=direction[...,0,0]
        elif size==2:
            # Outside the saved-adjugate region, this polynomial avoids a
            # factorization while preserving the full 2x2 curvature.
            tangent=(direction[...,0,0]*primal[...,1,1]+primal[...,0,0]*direction[...,1,1]
                     -direction[...,0,1]*primal[...,1,0]-primal[...,0,1]*direction[...,1,0])
        else:
            solved=_linalg.solve(primal,direction)
            tangent=output*solved.diagonal(dim1=-2,dim2=-1).sum(-1)
        return _autograd.ForwardTensor(output,tangent)
    layout=primal._data;needs_a=_state._grad_enabled and primal.requires_grad
    needs_e=_state._grad_enabled and direction.requires_grad
    source=layout.copy(order='K') if needs_a or needs_e else layout
    empty_layout=layout if not layout.size else None
    value=output._data;transposed=layout.flags.c_contiguous
    def matrix_backward(g):
        if not source.size:return _layout.gradient_copy(np.zeros_like(source),empty_layout)
        if source.shape[-1]==1:
            seed=np.asarray(g)
            return _layout.metadata_view(seed,source.shape,seed.strides+(seed.itemsize,seed.itemsize))
        return _diagonal_solve(source,g*value,factor_transposed=transposed,regularize=True)
    # JVP follows native 2.14.1's saved-LU reverse rule. The ordinary
    # det API retains its established singular-cofactor extension.
    output=_autograd.result(value,(primal,matrix_backward))
    with np.errstate(all='ignore'):
        solved=_solve_values(source,direction._data,False,factor_transposed=transposed,regularize=True)
        trace=_component(np.trace(solved,axis1=-2,axis2=-1));values=_component(trace*value)
    def direction_backward(g):
        return _diagonal_solve(source,g*value,factor_transposed=transposed,regularize=True)
    tangent=_autograd.result(values,(output,lambda g:g*trace),(direction,direction_backward),versioned=(output,))
    # Native exposes a primal alias; its retained gradient/hooks are separate
    # from the private determinant value consumed by the tangent graph.
    exposed=_autograd.view_result(output._data,output,lambda g:g)
    return _autograd.ForwardTensor(exposed,_layout.matching_tangent(tangent,exposed))


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


from . import _autograd, _errors, _layout, _linalg, _state, _tensor
