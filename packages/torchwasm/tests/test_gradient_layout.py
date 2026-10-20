"""Independent strided training and gradient ownership contracts for CPU/WASM."""
import itertools
import numpy as np
import pytest
from torchwasm import torch


@pytest.mark.parametrize('axes',list(itertools.permutations(range(3))))
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_dense_training_retains_layout_buffers_and_seed_ownership(axes,dtype):
    initial=np.arange(24,dtype=dtype).reshape(2,3,4).transpose(axes)/16
    x=torch.from_numpy(initial.copy(order='K')).requires_grad_();expected=initial.copy()
    optimizer=torch.optim.SGD([x],lr=.125);old=None
    for _ in range(4):
        optimizer.zero_grad(set_to_none=False)
        loss=x.square().sum();loss.backward()
        assert x.grad.stride()==x.stride()
        if old is not None:assert x.grad is old
        old=x.grad
        np.testing.assert_array_equal(x.grad.numpy(),expected*2)
        expected=(expected-expected*.25).astype(dtype)
        optimizer.step()
        np.testing.assert_array_equal(x.detach().numpy(),expected)
    seed=torch.full(x.shape,.5,dtype=getattr(torch,dtype));x.grad=None;x.backward(seed)
    x.grad.fill_(12);np.testing.assert_array_equal(seed.numpy(),np.full(x.shape,.5))


def test_half_dense_leaf_copies_seed_into_its_layout():
    x=torch.from_numpy(np.arange(24,dtype='float16').reshape(2,3,4).transpose(1,2,0)).requires_grad_()
    seed=torch.full(x.shape,.5,dtype=torch.float16);x.backward(seed)
    assert x.grad.stride()==x.stride()
    seed.fill_(2);np.testing.assert_array_equal(x.grad.numpy(),np.full(x.shape,.5))


def test_nondense_leaf_uses_contiguous_gradient_and_does_not_write_input_storage():
    base=np.arange(48,dtype='float64').reshape(4,12)
    x=torch.from_numpy(base[::2,::3]).requires_grad_();before=base.copy()
    x.square().sum().backward()
    assert x.grad.stride()==(4,1)
    np.testing.assert_array_equal(x.grad.numpy(),2*base[::2,::3])
    x.grad.fill_(99);np.testing.assert_array_equal(base,before)


@pytest.mark.parametrize('shape',[(),(1,),(0,3),(2,0,3),(1,0,3)])
def test_retained_gradients_have_contiguous_layout_and_preserve_previous_snapshot(shape):
    x=torch.ones(shape,requires_grad=True);middle=x*2;middle.retain_grad()
    seed=torch.ones(shape);middle.backward(seed,retain_graph=True);previous=middle.grad
    assert previous.is_contiguous()
    # Retained non-leaf accumulation replaces its buffer, unlike leaf accumulation.
    middle.backward(seed);assert middle.grad is not previous
    np.testing.assert_array_equal(previous.numpy(),np.ones(shape))
    np.testing.assert_array_equal(middle.grad.numpy(),np.full(shape,2.))
    expected=[];stride=1
    for size in reversed(shape):expected.append(stride);stride*=max(size,1)
    assert previous.stride()==tuple(reversed(expected))
    assert middle.grad.stride()==previous.stride()


def test_hook_gradient_and_post_accumulation_layout_are_independent():
    x=torch.from_numpy(np.array([[1.,2.,3.],[4.,5.,6.]],order='F')).requires_grad_()
    replacement=torch.full(x.shape,3.,dtype=torch.float64);seen=[]
    x.register_hook(lambda _:replacement)
    x.register_post_accumulate_grad_hook(lambda p:seen.append(p.grad.stride()))
    x.sum().backward();assert seen==[x.stride()]
    x.grad.add_(10);np.testing.assert_array_equal(replacement.numpy(),np.full(x.shape,3.))
