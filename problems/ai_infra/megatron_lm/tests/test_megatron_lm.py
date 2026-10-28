import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.ai_infra.megatron_lm.experiment import run

def fixture():
    return [torch.sin(torch.arange(n,dtype=torch.float64)*.3).reshape(shape) for n,shape in [(6,(2,3)),(12,(3,4)),(4,(4,)),(8,(4,2)),(2,(2,))]]

def scalar_oracle(x,a,c,b,d):
    out=[]
    for row in x.tolist():
        z=[sum(row[i]*a[i,j].item() for i in range(len(row)))+c[j].item() for j in range(len(c))]
        h=[.5*v*(1+math.tanh(math.sqrt(2/math.pi)*(v+.044715*v**3))) for v in z]
        out.append([sum(h[j]*b[j,o].item() for j in range(len(h)))+d[o].item() for o in range(len(d))])
    return x.new_tensor(out)

@pytest.mark.t0
def test_contract(impl):
    """Hidden dimensions must shard evenly and outputs retain the floating dtype."""
    args=fixture(); out=impl.parallel_mlp(*args,2)
    assert out.shape==(2,2) and out.dtype==torch.float64
    for bad in (0,3,True):
        with pytest.raises(ValueError): impl.parallel_mlp(*args,bad)

@pytest.mark.t1
def test_independent_scalar_forward(impl):
    """Scalar sums verify column/row pairing, local GeLU, and one output bias."""
    args=fixture(); before=[x.clone() for x in args]
    close(impl.parallel_mlp(*args,2),scalar_oracle(*args),atol=1e-12,rtol=1e-12)
    for a,b in zip(args,before): close(a,b)

@pytest.mark.t1
def test_bias_only(impl):
    """Zero hidden activations leave b2 once, irrespective of world size."""
    x,a,c,b,d=fixture(); x=x*0; c=c*0
    close(impl.parallel_mlp(x,a,c,b,d,4),d[None].expand(2,2))

@pytest.mark.t2
def test_shard_count_and_hidden_permutation(impl):
    """A hidden-channel permutation paired across both weights cannot change the function."""
    x,a,c,b,d=fixture(); p=torch.tensor([2,0,3,1]); expected=scalar_oracle(x,a,c,b,d)
    for parts in (1,2,4): close(impl.parallel_mlp(x,a[:,p],c[p],b[p],d,parts),expected,atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_all_gradients_finite_difference(impl):
    """Numerical scalar-oracle derivatives cover both weight matrices, biases and input."""
    args=[t.requires_grad_() for t in fixture()]; grads=torch.autograd.grad(impl.parallel_mlp(*args,2).sum(),args)
    for which in range(5):
        expected=torch.zeros_like(args[which])
        for j in range(args[which].numel()):
            plus=[a.detach().clone() for a in args]; minus=[a.detach().clone() for a in args]
            plus[which].reshape(-1)[j]+=1e-6; minus[which].reshape(-1)[j]-=1e-6
            expected.reshape(-1)[j]=(scalar_oracle(*plus).sum()-scalar_oracle(*minus).sum())/2e-6
        close(grads[which],expected,atol=1e-8,rtol=1e-6)

@pytest.mark.t3
def test_tensor_parallel_training(impl,artifact):
    """Training a sharded output projection follows dense SGD at every step."""
    result=run(impl.parallel_mlp); artifact('parallel_training',result)
    assert result['metrics']['trajectory_error']<1e-12
    assert result['metrics']['final_loss']<result['metrics']['initial_loss']*.1
