import math
import random
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.graphsage_mean.toy import scalar_mean, scalar_encode, graph
from problems.representation.graphsage_mean.experiment import run


def fixture(dtype=torch.float64):
    rng=random.Random(1729)
    x=torch.tensor([[rng.uniform(-.8,1.2) for _ in range(3)] for _ in range(4)],dtype=dtype)
    weights=[torch.tensor([[rng.uniform(-.7,.9) for _ in range(out)] for _ in range(inp)],dtype=dtype) for inp,out in ((6,4),(8,3))]
    ids=[torch.tensor(v,dtype=torch.long) for v in [[[1,1,99],[0,2,3],[-99,99,-1],[1,2,-99]],[[3,2,-99],[2,2,0],[1,3,-99],[0,0,-99]]]]
    masks=[torch.tensor(v,dtype=torch.bool) for v in [[[True,True,False],[True,True,True],[False,False,False],[True,True,False]],[[True,True,False],[True,True,True],[True,True,False],[True,True,False]]]]
    return x,ids,masks,weights


@pytest.mark.t0
def test_mean_contract_padding_empty_samples_and_validation(impl):
    for dtype in (torch.float32,torch.float64):
        for nodes,samples in ((0,2),(3,0),(3,2)):
            x=torch.randn(nodes,6,dtype=dtype)[:,::2];ids=torch.full((nodes,samples),99,dtype=torch.long);mask=torch.zeros(nodes,samples,dtype=torch.bool)
            before=x.clone();result=impl.masked_neighbor_mean(x,ids,mask)
            assert result.shape==(nodes,3) and result.dtype==dtype and result.device==x.device
            close(result,torch.zeros(nodes,3,dtype=dtype));close(x,before)
    x=torch.zeros(3,2);ids=torch.zeros(3,2,dtype=torch.long);mask=torch.ones(3,2,dtype=torch.bool)
    for args in [(x.long(),ids,mask),(torch.zeros(3,0),ids,mask),(x,ids.float(),mask),(x,ids,mask.long()),(x,ids[:2],mask[:2]),(x,ids,mask[:2]),(x+math.nan,ids,mask),(x,ids+3,mask),(x,ids-1,mask)]:
        with pytest.raises(ValueError):impl.masked_neighbor_mean(*args)


@pytest.mark.t0
def test_encoder_contract_precision_empty_batch_identity_and_validation(impl):
    for dtype in (torch.float32,torch.float64):
        for nodes in (0,3):
            x=torch.randn(nodes,6,dtype=dtype)[:,::2];ids=torch.zeros(nodes,0,dtype=torch.long);mask=torch.zeros(nodes,0,dtype=torch.bool);w=torch.randn(6,8,dtype=dtype)[:,::2]
            close(impl.graphsage_encode(x,[],[],[]),x)
            result=impl.graphsage_encode(x,[ids],[mask],[w]);assert result.shape==(nodes,4) and result.dtype==dtype and result.device==x.device
    x,ids,masks,weights=fixture()
    for eps in (0.,-1.,math.nan,math.inf,True):
        with pytest.raises(ValueError):impl.graphsage_encode(x,ids,masks,weights,eps)
    for index,mask,ws in [(ids[:1],masks,weights),(ids,masks,weights[:1]),(ids,masks,[torch.zeros(5,4,dtype=x.dtype),weights[1]]),(ids,masks,[weights[0].float(),weights[1]]),(ids,masks,[weights[0]+math.inf,weights[1]]),(ids,masks,[weights[0],torch.zeros(7,3,dtype=x.dtype)])]:
        with pytest.raises(ValueError):impl.graphsage_encode(x,index,mask,ws)


@pytest.mark.t1
def test_hand_mean_concatenation_relu_and_feature_axis_normalization(impl):
    x=torch.tensor([[1.,2.],[-3.,4.],[5.,-2.]],dtype=torch.float64)
    ids=torch.tensor([[1,1,99],[0,2,-99],[-1,99,99]]);mask=torch.tensor([[True,True,False],[True,True,False],[False,False,False]])
    close(impl.masked_neighbor_mean(x,ids,mask),torch.tensor([[-3.,4.],[3.,0.],[0.,0.]],dtype=x.dtype),atol=1e-12,rtol=1e-12)
    w=torch.tensor([[1.,-1.,.2],[.5,.3,-.7],[-.4,.8,.6],[.2,-.5,.4]],dtype=x.dtype)
    expected=torch.tensor([[1.,0.,0.],[0.,1.,0.],[4/math.sqrt(21.76),0.,2.4/math.sqrt(21.76)]],dtype=x.dtype)
    close(impl.graphsage_encode(x,[ids],[mask],[w]),expected,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_independent_scalar_multihop_and_input_preservation(impl):
    for dtype in (torch.float32,torch.float64):
        x,ids,masks,weights=fixture(dtype);before=[v.clone() for v in [x,*ids,*masks,*weights]]
        for depth in (0,1,2):
            expected=scalar_encode(x.tolist(),[v.tolist() for v in ids[:depth]],[v.tolist() for v in masks[:depth]],[v.tolist() for v in weights[:depth]])
            tolerance=4e-6 if dtype==torch.float32 else 1e-11
            close(impl.graphsage_encode(x,ids[:depth],masks[:depth],weights[:depth]),torch.tensor(expected,dtype=dtype),atol=tolerance,rtol=tolerance)
        for value,old in zip([x,*ids,*masks,*weights],before):close(value,old)


@pytest.mark.t2
def test_strided_sampling_weights_and_neighbor_order(impl):
    for dtype in (torch.float32,torch.float64):
        x,ids,masks,weights=fixture(dtype)
        def strided(t):
            expanded=torch.stack([t,torch.zeros_like(t)],dim=-1).reshape(*t.shape[:-1],2*t.shape[-1]);return expanded[...,::2]
        x=strided(x);ids=[strided(v).to(torch.int32) for v in ids];masks=[strided(v) for v in masks];weights=[strided(v) for v in weights]
        expected=torch.tensor(scalar_encode(x.tolist(),[v.tolist() for v in ids],[v.tolist() for v in masks],[v.tolist() for v in weights]),dtype=dtype)
        tolerance=4e-6 if dtype==torch.float32 else 1e-11
        close(impl.graphsage_encode(x,ids,masks,weights),expected,atol=tolerance,rtol=tolerance)
        order=torch.tensor([2,0,1]);close(impl.graphsage_encode(x,[v[:,order] for v in ids],[v[:,order] for v in masks],weights),expected,atol=tolerance,rtol=tolerance)


@pytest.mark.t2
def test_repeated_neighbor_gradient_counts_and_empty_context_paths(impl):
    x=torch.tensor([[1.,2.],[-3.,4.],[5.,-2.]],dtype=torch.float64,requires_grad=True)
    ids=torch.tensor([[1,1,99],[0,2,-99],[-1,99,99]]);mask=torch.tensor([[True,True,False],[True,True,False],[False,False,False]])
    coeff=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=x.dtype)
    (impl.masked_neighbor_mean(x,ids,mask)*coeff).sum().backward()
    close(x.grad,torch.tensor([[1.5,2.],[1.,2.],[1.5,2.]],dtype=x.dtype),atol=1e-12,rtol=1e-12)
    empty=torch.randn(3,2,dtype=x.dtype,requires_grad=True)
    impl.masked_neighbor_mean(empty,torch.empty(3,0,dtype=torch.long),torch.empty(3,0,dtype=torch.bool)).sum().backward()
    close(empty.grad,torch.zeros_like(empty))


@pytest.mark.t2
def test_feature_and_every_layer_gradient_against_scalar_finite_differences(impl):
    x,ids,masks,weights=fixture();values=[x.tolist(),*[w.tolist() for w in weights]]
    x=x.requires_grad_();weights=[w.requires_grad_() for w in weights]
    coeff=[[.2,-.5,.7],[-.3,.4,.1],[.8,.2,-.6],[-.4,.9,.3]]
    result=impl.graphsage_encode(x,ids,masks,weights);(result*torch.tensor(coeff,dtype=x.dtype)).sum().backward()
    def objective():
        output=scalar_encode(values[0],[v.tolist() for v in ids],[v.tolist() for v in masks],values[1:])
        return math.fsum(a*b for row,crow in zip(output,coeff) for a,b in zip(row,crow))
    for matrix,tensor in zip(values,[x,*weights]):
        assert tensor.grad is not None,'All features and layer weights must remain differentiable'
        expected=[];step=1e-6
        for row in matrix:
            derivatives=[]
            for column,value in enumerate(row):
                row[column]=value+step;plus=objective();row[column]=value-step;minus=objective();row[column]=value
                derivatives.append((plus-minus)/(2*step))
            expected.append(derivatives)
        close(tensor.grad,torch.tensor(expected,dtype=x.dtype),atol=2e-6,rtol=2e-6)


@pytest.mark.t2
def test_node_relabeling_equivariance_and_shared_weights_on_unseen_graphs(impl):
    x,ids,masks,weights=fixture();wanted=impl.graphsage_encode(x,ids,masks,weights)
    order=torch.tensor([2,0,3,1]);inverse=torch.tensor([1,3,0,2])
    remapped=[]
    for index,mask in zip(ids,masks):
        safe=torch.where(mask,index,torch.zeros_like(index));remapped.append(inverse[safe][order])
    actual=impl.graphsage_encode(x[order],remapped,[m[order] for m in masks],weights)
    close(actual,wanted[order],atol=1e-12,rtol=1e-12)
    for seed,nodes in [(17,14),(83,18)]:
        h,index,valid,_,_=graph(seed,queries=8 if nodes==14 else 10,anchors=6 if nodes==14 else 8)
        ws=[torch.ones(6,5,dtype=h.dtype)*.2,torch.ones(10,4,dtype=h.dtype)*.1]
        assert impl.graphsage_encode(h,index,valid,ws).shape==(nodes,4)


@pytest.mark.t2
def test_normalize_each_hop_not_only_the_last(impl):
    x,ids,masks,weights=fixture();first=impl.graphsage_encode(x,ids[:1],masks[:1],weights[:1]);second=impl.graphsage_encode(first,ids[1:],masks[1:],weights[1:])
    close(impl.graphsage_encode(x,ids,masks,weights),second,atol=1e-12,rtol=1e-12)
    expected=torch.tensor(scalar_encode(x.tolist(),[v.tolist() for v in ids],[v.tolist() for v in masks],[v.tolist() for v in weights]),dtype=x.dtype)
    close(second,expected,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_zero_relu_eps_boundary_and_empty_node_gradients(impl):
    for dtype in (torch.float32,torch.float64):
        x=torch.tensor([[1.,0.],[0.,0.]],dtype=dtype,requires_grad=True);index=torch.zeros(2,0,dtype=torch.long);mask=torch.zeros(2,0,dtype=torch.bool)
        weight=torch.tensor([[1e-9,-1.],[0.,-1.],[0.,0.],[0.,0.]],dtype=dtype,requires_grad=True)
        output=impl.graphsage_encode(x,[index],[mask],[weight],eps=1e-6)
        close(output,torch.tensor([[.001,0.],[0.,0.]],dtype=dtype),atol=1e-7,rtol=1e-6)
        output.sum().backward();assert bool(torch.isfinite(x.grad).all()) and bool(torch.isfinite(weight.grad).all())
        empty=torch.empty(0,2,dtype=dtype,requires_grad=True);w=torch.ones(4,3,dtype=dtype,requires_grad=True)
        impl.graphsage_encode(empty,[torch.empty(0,2,dtype=torch.long)],[torch.empty(0,2,dtype=torch.bool)],[w]).sum().backward()
        close(w.grad,torch.zeros_like(w))


@pytest.mark.t3
def test_actual_training_and_inductive_held_out_queries(impl):
    metrics=run(impl.graphsage_encode)['metrics']
    assert metrics['updates']==100 and metrics['train_nodes']!=metrics['held_out_nodes']
    assert metrics['trainable_parameters']==80 and metrics['feature_only_accuracy_ceiling']==.5
    assert metrics['final_loss']<metrics['initial_loss']*.3
    assert metrics['final_accuracy']>=.9 and metrics['held_out_accuracy']>=.9
    assert metrics['held_out_loss']<.3
