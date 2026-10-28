import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.ai_infra.gptq.experiment import run

def schur_oracle(w,u,scales,lo,hi):
    # Independent inverse-Hessian elimination, without using the Cholesky update.
    d=len(u); h=[[sum(u[k][i]*u[k][j] for k in range(d)) for j in range(d)] for i in range(d)]
    rows=[list(row) for row in w]; out=[[0.]*d for row in rows]
    for j in range(d):
        pivot=h[j][j]
        for r,row in enumerate(rows):
            quant=max(lo,min(hi,round(row[j]/scales[r])))*scales[r]; out[r][j]=quant; err=row[j]-quant
            for k in range(j+1,d): row[k]-=err*h[j][k]/pivot
        old=[row[:] for row in h]
        for a in range(j+1,d):
            for b in range(j+1,d): h[a][b]=old[a][b]-old[a][j]*old[j][b]/pivot
    return out

@pytest.mark.t0
def test_contract(impl):
    """The supplied inverse-Hessian factor is upper triangular with a positive diagonal."""
    w=torch.ones(2,3,dtype=torch.float64); u=torch.eye(3,dtype=torch.float64); s=torch.ones(2,dtype=torch.float64)
    assert impl.gptq_quantize(w,u,s).shape==w.shape
    u[2,0]=.1
    with pytest.raises(ValueError): impl.gptq_quantize(w,u,s)
    with pytest.raises(ValueError): impl.gptq_quantize(w,torch.eye(3,dtype=torch.float64),s*0)

@pytest.mark.t1
def test_hand_compensation(impl):
    """Quantizing .49 to zero shifts .70 by -.392 before its own rounding."""
    w=torch.tensor([[.49,.7]],dtype=torch.float64); u=torch.tensor([[1.,.8],[0.,.6]],dtype=torch.float64)
    close(impl.gptq_quantize(w,u,torch.ones(1,dtype=torch.float64)),torch.zeros_like(w))

@pytest.mark.t1
def test_independent_schur_complement(impl):
    """Inverse-Hessian elimination independently reproduces sequential compensation for multiple rows."""
    w=torch.tensor([[.49,.7,-.2],[-.42,.31,.8]],dtype=torch.float64); u=torch.tensor([[1.3,.8,-.3],[0.,.7,.2],[0.,0.,1.1]],dtype=torch.float64); s=torch.tensor([.4,.3],dtype=torch.float64)
    before=w.clone(); expected=w.new_tensor(schur_oracle(w.tolist(),u.tolist(),s.tolist(),-2,2))
    close(impl.gptq_quantize(w,u,s,-2,2),expected); close(w,before)

@pytest.mark.t2
def test_diagonal_factor_reduces_to_rounding(impl):
    """With independent calibration channels, compensation cannot change later columns."""
    w=torch.tensor([[-3.2,-.5,.5,1.5,4.]],dtype=torch.float64); u=torch.eye(5,dtype=torch.float64)*2
    close(impl.gptq_quantize(w,u,torch.ones(1,dtype=torch.float64),-2,2),torch.tensor([[-2.,0.,0.,2.,2.]],dtype=torch.float64))

@pytest.mark.t2
def test_factor_scale_and_offline_gradient(impl):
    """Scaling the whole inverse-Hessian factor cancels in its row-to-diagonal ratios; PTQ is detached."""
    w=torch.tensor([[.49,.7]],dtype=torch.float64,requires_grad=True); u=torch.tensor([[1.,.8],[0.,.6]],dtype=torch.float64); s=torch.ones(1,dtype=torch.float64)
    a=impl.gptq_quantize(w,u,s); b=impl.gptq_quantize(w,u*7,s)
    close(a,b); assert not a.requires_grad

@pytest.mark.t3
def test_correlated_reconstruction(impl,artifact):
    """Second-order correction lowers output error on the declared correlated two-feature fixture."""
    r=run(impl.gptq_quantize); artifact('gptq_calibration',r)
    assert r['metrics']['compensated_mse']<r['metrics']['rtn_mse']*.5
