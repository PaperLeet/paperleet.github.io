import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.retrieval.semantic_ids.toy import data, scalar_quantize
from problems.retrieval.semantic_ids.experiment import run


@pytest.mark.t0
def test_shapes_precision_frozen_strided_and_empty(impl):
    """Frozen encoding preserves precision/device, int64 indices, strided values and empty batches."""
    for dtype in (torch.float32,torch.float64):
        for batch in (0,1,4):
            x=torch.randn(batch,6,dtype=dtype,requires_grad=True)[:,::2]
            books=torch.randn(2,5,6,dtype=dtype,requires_grad=True)[...,::2]
            a,b=x.detach().clone(),books.detach().clone()
            codes,reconstruction=impl.residual_quantize(x,books)
            assert codes.shape==(batch,2) and codes.dtype==torch.long
            assert reconstruction.shape==x.shape and reconstruction.dtype==dtype
            assert codes.device==x.device and reconstruction.device==x.device
            assert not codes.requires_grad and not reconstruction.requires_grad
            ids=impl.resolve_collisions(codes)
            assert ids.shape==(batch,3) and ids.dtype==torch.long and ids.device==codes.device
            close(x,a);close(books,b)


@pytest.mark.t1
def test_hand_computed_residual_levels(impl):
    """For z=4.8 choose 4, then .75 from residual .8; reconstruct 4.75."""
    x=torch.tensor([[4.8],[-.8]],dtype=torch.float64)
    books=torch.tensor([[[0.],[4.],[-4.]],[[0.],[.75],[-.75]]],dtype=x.dtype)
    codes,value=impl.residual_quantize(x,books)
    assert codes.tolist()==[[1,1],[0,2]]
    close(value,torch.tensor([[4.75],[-.75]],dtype=x.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_independent_scalar_oracle_rectangular_shapes(impl):
    """Each distinct level quantizes its remaining residual; Python distances are an independent oracle."""
    for batch,levels,count,features in [(3,1,4,2),(5,3,7,4),(2,4,1,3)]:
        x,books=torch.randn(batch,features,dtype=torch.float64),torch.randn(levels,count,features,dtype=torch.float64)
        codes,value=impl.residual_quantize(x,books)
        wanted,expected=scalar_quantize(x.tolist(),books.tolist())
        assert codes.tolist()==wanted
        close(value,torch.tensor(expected,dtype=x.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_exact_ties_choose_smallest_index(impl):
    """Squared-distance ties choose the lowest code index at each level."""
    x=torch.tensor([[0.,0.],[2.,0.]],dtype=torch.float64)
    books=torch.tensor([[[-1.,0.],[1.,0.],[1.,0.]],[[0.,-1.],[0.,1.],[0.,1.]]],dtype=x.dtype)
    codes,value=impl.residual_quantize(x,books)
    assert codes.tolist()==[[0,0],[1,0]]
    close(value,torch.tensor([[-1.,-1.],[1.,-1.]],dtype=x.dtype))


@pytest.mark.t2
def test_reconstruction_scaling_and_row_independence(impl):
    """Reconstruction sums all levels; positive scaling and batch permutation preserve selected codes."""
    x,books=data();codes,value=impl.residual_quantize(x,books)
    expected=torch.zeros_like(x)
    for level,book in enumerate(books): expected=expected+book[codes[:,level]]
    close(value,expected,atol=1e-12,rtol=1e-12)
    scaled,scaled_value=impl.residual_quantize(8*x,8*books)
    assert scaled.tolist()==codes.tolist()
    close(scaled_value,8*value)
    order=torch.tensor([7,2,0,6,1,3,5,4],dtype=torch.long)
    shuffled,shuffled_value=impl.residual_quantize(x[order],books)
    assert shuffled.tolist()==codes[order].tolist()
    close(shuffled_value,value[order])


@pytest.mark.t2
def test_collision_groups_are_local_stable_and_noncontiguous(impl):
    """Equal full tuples alone collide, each with its own stable occurrence counter."""
    codes=torch.tensor([[7,1,4],[8,2,0],[7,1,4],[7,2,4],[8,2,0],[7,1,4]],dtype=torch.long)
    before=codes.clone();actual=impl.resolve_collisions(codes)
    assert actual.tolist()==[[7,1,4,0],[8,2,0,0],[7,1,4,1],[7,2,4,0],[8,2,0,1],[7,1,4,2]]
    assert len(set(map(tuple,actual.tolist())))==6
    close(codes,before)
    assert impl.resolve_collisions(torch.tensor([[9],[9],[2]],dtype=torch.long)).tolist()==[[9,0],[9,1],[2,0]]


@pytest.mark.t2
def test_zero_available_gives_nonincreasing_residual_error(impl):
    """Only with zero in every codebook is nonincreasing residual error guaranteed."""
    x,books=data();previous=x.square().sum().item()
    for depth in (1,2,3):
        _,value=impl.residual_quantize(x,books[:depth]);error=(x-value).square().sum().item()
        assert error<=previous+1e-12
        previous=error


@pytest.mark.t3
def test_frozen_catalog_experiment(impl,artifact):
    """Learner quantization matches an independent scalar oracle and disambiguates eight synthetic items."""
    result=run(impl.residual_quantize,impl.resolve_collisions);artifact('semantic_id_experiment',result)
    metrics=result['metrics']
    assert metrics['maximum_scalar_error']<1e-12 and metrics['suffix_error']==0
    assert metrics['unique_full_ids']==8 and metrics['semantic_prefixes']==6
    assert metrics['final_rmse']<metrics['initial_rmse']
