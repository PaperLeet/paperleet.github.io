import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from ..toy import fixture, normalized, scalar_loss
from ..experiment import run


@pytest.mark.t0
def test_precision_strides_shapes_and_inputs_unchanged(impl):
    """Loss/EMA/queue preserve float32/64, support strided rows and do not modify inputs."""
    for dtype in (torch.float32,torch.float64):
        q=(torch.arange(24,dtype=dtype).reshape(4,6)+.3)[:,::2];k=q.flip(0);queue=k[:3].clone()
        before=[v.clone() for v in (q,k,queue)]
        loss=impl.moco_loss(q,k,queue,.4);assert loss.shape==() and loss.dtype==dtype and loss.device==q.device
        close(loss,torch.tensor(scalar_loss(q.tolist(),k.tolist(),queue.tolist(),.4),dtype=dtype),atol=2e-6,rtol=2e-6)
        new_key=impl.momentum_update(q,k,.8);updated,pointer=impl.enqueue(queue,k,2)
        assert new_key.shape==q.shape and updated.shape==queue.shape and updated.dtype==dtype and type(pointer) is int
        for value,original in zip((q,k,queue),before):close(value,original)


@pytest.mark.t0
def test_invalid_input_contract(impl):
    """Reject mismatched/nonfinite tensors, invalid scalar coefficients and invalid pointers."""
    q=torch.ones(2,3,dtype=torch.float64);queue=q.clone()
    for x,y,z in [(q[:0],q[:0],queue),(q,q[:1],queue),(q,q,queue[:0]),(q.flatten(),q,queue),
                  (q[:,:0],q[:,:0],queue[:,:0]),(q,q.float(),queue),(q.long(),q.long(),queue.long()),
                  (q*float('nan'),q,queue),(q,q,queue*float('inf'))]:
        with pytest.raises(ValueError):impl.moco_loss(x,y,z)
    for value in (0,-1,True,'0.2',torch.tensor(.2),float('inf'),float('nan')):
        with pytest.raises(ValueError):impl.moco_loss(q,q,queue,value)
    for value in (-.1,1.1,True,'0.9',torch.tensor(.9),float('nan')):
        with pytest.raises(ValueError):impl.momentum_update(q,q,value)
    for a,b in ((q,q[:1]),(q,q.float()),(q.long(),q.long()),(q,q*float('inf'))):
        with pytest.raises(ValueError):impl.momentum_update(a,b)
    for pointer in (-1,2,True,.5):
        with pytest.raises(ValueError):impl.enqueue(queue,q,pointer)
    for a,b in ((queue[:0],q),(queue,q.float()),(queue,q[:,:2]),(queue,q*float('nan'))):
        with pytest.raises(ValueError):impl.enqueue(a,b,0)


@pytest.mark.t1
def test_one_positive_plus_dictionary_hand_computation(impl):
    """The denominator includes its positive and K queue negatives, never the other batch queries."""
    q=torch.tensor([[1.,0.]],dtype=torch.float64);queue=torch.tensor([[-1.,0.],[0.,1.]],dtype=q.dtype)
    for t in (.1,.7,2.):
        expected=math.log1p(math.exp(-2/t)+math.exp(-1/t))
        close(impl.moco_loss(q,q,queue,t),torch.tensor(expected,dtype=q.dtype),atol=1e-12,rtol=1e-12)
    collapsed=torch.tensor([[2.,-1.]]*3,dtype=q.dtype)
    close(impl.moco_loss(collapsed,collapsed,collapsed,.3),torch.tensor(math.log(4),dtype=q.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t1
def test_momentum_direction_and_circular_wrap_hand_computation(impl):
    """EMA weights old key by m; queue writes begin at pointer and overwrite oldest slots."""
    q=torch.tensor([2.,-4.],dtype=torch.float64);k=torch.tensor([-2.,8.],dtype=q.dtype)
    close(impl.momentum_update(q,k,.75),torch.tensor([-1.,5.],dtype=q.dtype),atol=1e-12,rtol=0)
    queue=torch.eye(3,dtype=q.dtype);keys=torch.tensor([[3.,4.,0.],[0.,0.,-2.]],dtype=q.dtype)
    actual,pointer=impl.enqueue(queue,keys,2)
    close(actual,torch.tensor([[0.,0.,-1.],[0.,1.,0.],[.6,.8,0.]],dtype=q.dtype),atol=1e-12,rtol=0);assert pointer==1


@pytest.mark.t2
def test_query_finite_difference_and_no_key_or_dictionary_gradients(impl):
    """Only queries differentiate, including through row normalization; keys/queue receive no graph."""
    q=torch.tensor([[.4,-.7,1.2],[1.3,.8,-.2]],dtype=torch.float64,requires_grad=True)
    k=torch.tensor([[1.1,.2,-.3],[-.4,1.8,.3]],dtype=q.dtype,requires_grad=True)
    queue=torch.tensor([[.2,.9,.8],[-.7,.3,.6],[1.,-.5,.4]],dtype=q.dtype,requires_grad=True)
    dq,dk,dqueue=torch.autograd.grad(impl.moco_loss(q,k,queue,.37),(q,k,queue),allow_unused=True)
    assert dk is None and dqueue is None
    rows=q.detach().tolist();numeric=[];h=1e-6
    for i,row in enumerate(rows):
        numeric.append([])
        for j,value in enumerate(row):
            row[j]=value+h;plus=scalar_loss(rows,k.tolist(),queue.tolist(),.37)
            row[j]=value-h;minus=scalar_loss(rows,k.tolist(),queue.tolist(),.37);row[j]=value
            numeric[-1].append((plus-minus)/(2*h))
    close(dq,torch.tensor(numeric,dtype=q.dtype),atol=2e-8,rtol=2e-6)


@pytest.mark.t2
def test_pair_queue_feature_permutations_scale_and_batch_mean(impl):
    """Paired order, dictionary order, orthogonal basis and independent positive row scales preserve loss."""
    a,b,w,queue=fixture();q=a[:3,:3];k=b[:3,:3];base=impl.moco_loss(q,k,queue,.4)
    order=torch.tensor([2,0,1]);negative_order=torch.tensor([4,1,3,0,2]);feature=torch.tensor([2,0,1]);sign=torch.tensor([-1.,1.,-1.],dtype=q.dtype)
    close(impl.moco_loss(q[order],k[order],queue[negative_order],.4),base,atol=1e-12,rtol=1e-12)
    close(impl.moco_loss(q[:,feature]*sign,k[:,feature]*sign,queue[:,feature]*sign,.4),base,atol=1e-12,rtol=1e-12)
    scales=torch.tensor([[.1],[3.],[100.]],dtype=q.dtype)
    close(impl.moco_loss(q*scales,k/scales,queue*17,.4),base,atol=1e-12,rtol=1e-12)
    individual=torch.stack([impl.moco_loss(q[i:i+1],k[i:i+1],queue,.4) for i in range(3)]).mean()
    close(individual,base,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_scalar_denominator_across_temperatures_and_zero_rows(impl):
    """Independent stable scalar logits cover zero/tiny rows, large scales and extreme finite logits."""
    q=torch.tensor([[0.,0.],[1e-14,-2e-14],[3e10,-4e10]],dtype=torch.float64)
    k=torch.tensor([[2.,1.],[-2e-14,1e-14],[1e10,3e10]],dtype=q.dtype)
    queue=torch.tensor([[.3,-.8],[2.,3.],[-2.,1.]],dtype=q.dtype)
    for temperature in (1e-4,.3,1000.):
        actual=impl.moco_loss(q,k,queue,temperature);assert bool(torch.isfinite(actual))
        close(actual,torch.tensor(scalar_loss(q.tolist(),k.tolist(),queue.tolist(),temperature),dtype=q.dtype),atol=2e-9,rtol=1e-11)


@pytest.mark.t2
def test_momentum_endpoints_arbitrary_shape_detachment_and_independent_storage(impl):
    """m=0 copies query, m=1 freezes key; neither endpoint aliases inputs or keeps their gradients."""
    for shape in ((),(0,),(2,3)):
        q=torch.full(shape,2.,dtype=torch.float64,requires_grad=True);k=torch.full(shape,-3.,dtype=q.dtype,requires_grad=True)
        for m in (0.,.9,1.):
            out=impl.momentum_update(q,k,m);assert not out.requires_grad
            close(out,q.detach()*(1-m)+k.detach()*m,atol=1e-12,rtol=1e-12)
            out.fill_(19.);close(q,torch.full_like(q,2.));close(k,torch.full_like(k,-3.))


@pytest.mark.t2
def test_queue_empty_large_batches_multiple_wraps_and_detachment(impl):
    """Physical slot order and pointer match sequential FIFO for B=0, B=K and B>K; outputs are detached."""
    for capacity in (1,4):
        for batch in (0,1,capacity,capacity+2,2*capacity+3):
            for pointer in (0,capacity-1):
                queue=torch.arange(capacity*2,dtype=torch.float64).reshape(capacity,2)+.2
                keys=(torch.arange(batch*4,dtype=queue.dtype).reshape(batch,4)+1)[:,::2].requires_grad_();queue.requires_grad_()
                original_q,original_k=queue.clone(),keys.clone();expected=queue.detach().tolist()
                for i,row in enumerate(normalized(keys.tolist())):expected[(pointer+i)%capacity]=row
                out,next_pointer=impl.enqueue(queue,keys,pointer)
                assert next_pointer==(pointer+batch)%capacity and not out.requires_grad
                close(out,torch.tensor(expected,dtype=queue.dtype),atol=1e-12,rtol=1e-12)
                out[0,0]=99.;close(queue,original_q);close(keys,original_k)


@pytest.mark.t3
def test_actual_sgd_ema_queue_trajectory_matches_independent_scalar_dynamics(impl,artifact):
    """Twelve actual SGD steps update query weights, momentum keys and a five-slot dictionary independently."""
    result=run(impl.moco_loss,impl.momentum_update,impl.enqueue);artifact('moco_training',result);m=result['metrics']
    assert m['updates']==12 and m['trainable_parameters']==12 and m['dictionary_size']==5
    assert m['maximum_scalar_loss_error']<1e-8 and m['maximum_state_error']<1e-8
    assert m['maximum_query_gradient']>.01 and m['final_pointer']==4
