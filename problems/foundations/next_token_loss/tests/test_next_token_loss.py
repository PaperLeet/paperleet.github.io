import math
import pytest
import torch


def scalar_oracle(logits, tokens):
    rows, ids = logits.tolist(), tokens.tolist()
    terms = []
    for row, target in zip(rows, ids):
        for t in range(len(target)-1):
            m = max(row[t])
            terms.append(math.log(sum(math.exp(v-m) for v in row[t])) - (row[t][target[t+1]]-m))
    return sum(terms)/len(terms) if terms else 0.


@pytest.mark.t0
def test_contract(impl):
    """Return a scalar with the logits dtype and device."""
    for dtype in (torch.float32, torch.float64):
        x = torch.zeros(2, 3, 4, dtype=dtype)
        actual = impl.next_token_loss(x, torch.tensor([[0,1,2],[3,2,1]]))
        assert actual.shape == () and actual.dtype == dtype and actual.device == x.device
        torch.testing.assert_close(actual, torch.tensor(math.log(4), dtype=dtype))


@pytest.mark.t1
def test_independent_scalar_oracle(impl):
    """Compare unequal per-position scores against a scalar shifted-token NLL."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[[3.,-1.,2.],[0.,4.,-2.],[-2.,1.,3.],[5.,2.,-1.]],
                          [[-1.,2.,0.],[3.,0.,-2.],[1.,-3.,4.],[1.,2.,3.]]], dtype=dtype)
        y = torch.tensor([[0,2,1,0],[2,0,2,1]])
        torch.testing.assert_close(impl.next_token_loss(x,y), torch.tensor(scalar_oracle(x,y),dtype=dtype))


@pytest.mark.t2
def test_boundaries_and_extreme_scores(impl):
    """Cover one-token sequences, one-class vocabulary and extreme finite logits."""
    x = torch.tensor([[[1.,-1.]],[[3.,4.]]],dtype=torch.float64,requires_grad=True)
    zero = impl.next_token_loss(x,torch.tensor([[0],[1]])); zero.backward()
    torch.testing.assert_close(zero,torch.tensor(0.,dtype=x.dtype))
    torch.testing.assert_close(x.grad,torch.zeros_like(x))
    x = torch.tensor([[[10000.,-10000.],[-10000.,10000.],[5.,6.]]],dtype=torch.float64)
    torch.testing.assert_close(impl.next_token_loss(x,torch.tensor([[0,1,0]])),torch.tensor(20000.,dtype=x.dtype))
    torch.testing.assert_close(impl.next_token_loss(torch.ones(2,3,1),torch.zeros(2,3,dtype=torch.long)),torch.tensor(0.))


@pytest.mark.t2
def test_large_common_offset_preserves_loss_and_gradient(impl):
    """A large shared offset cannot erase log(V) or change softmax gradients."""
    for dtype in (torch.float32, torch.float64):
        x = torch.full((1, 2, 2), 1e20, dtype=dtype, requires_grad=True)
        loss = impl.next_token_loss(x, torch.tensor([[0, 1]]))
        torch.testing.assert_close(loss, torch.tensor(math.log(2), dtype=dtype))
        loss.backward()
        torch.testing.assert_close(x.grad, torch.tensor([[[.5, -.5], [0., 0.]]], dtype=dtype))


@pytest.mark.t2
def test_analytic_shifted_gradients(impl):
    """Only predicting positions receive (probability minus next-label one-hot)/count."""
    x = (torch.arange(24,dtype=torch.float64).reshape(2,4,3)/7).requires_grad_()
    y = torch.tensor([[2,0,1,2],[0,2,1,0]])
    impl.next_token_loss(x,y).backward()
    expected = torch.zeros_like(x)
    expected[:,:-1,:] = torch.softmax(x.detach()[:,:-1,:],-1)/6
    for b in range(2):
        for t in range(3): expected[b,t,int(y[b,t+1])] -= 1/6
    torch.testing.assert_close(x.grad,expected)


@pytest.mark.t2
def test_input_preservation_and_sequence_boundaries(impl):
    """Offsets, first labels and last logits cannot change the shifted objective."""
    x = torch.arange(48,dtype=torch.float64).reshape(2,4,6)[...,::2]/9
    y = torch.tensor([[0,1,2,0],[2,0,1,2]]); before=x.clone(); before_y=y.clone()
    value=impl.next_token_loss(x,y)
    shifted=x+torch.tensor([[[20.],[-4.],[3.],[-99.]],[[7.],[8.],[-2.],[35.]]],dtype=x.dtype)
    torch.testing.assert_close(impl.next_token_loss(shifted,y),value)
    altered=x.clone(); altered[:,-1,:]=10000.; labels=y.clone(); labels[:,0]=(labels[:,0]+1)%3
    torch.testing.assert_close(impl.next_token_loss(altered,labels),value)
    torch.testing.assert_close(impl.next_token_loss(x.repeat(3,1,1),y.repeat(3,1)),value)
    torch.testing.assert_close(x,before); torch.testing.assert_close(y,before_y)


@pytest.mark.t3
def test_prediction_confidence_experiment(impl,artifact):
    """Execute the learner loss on controlled next-token confidence margins."""
    from problems.foundations.next_token_loss.experiment import run
    artifact('toy_experiment',run(impl.next_token_loss))
