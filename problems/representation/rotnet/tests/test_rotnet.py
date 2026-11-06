import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from ..toy import fixture, scalar_ce
from ..experiment import run


@pytest.mark.t0
def test_shape_dtype_no_mutation_and_invalid_inputs(impl):
    """Four exact square-image rotations preserve floating data and generate int64 labels."""
    for dtype in (torch.float32,torch.float64):
        images=fixture().to(dtype);old=images.clone();views,labels=impl.rotation_batch(images)
        assert views.shape==(32,1,4,4) and views.dtype==dtype
        assert labels.dtype==torch.int64 and labels.shape==(32,)
        close(images,old,atol=0,rtol=0)
    for bad in (images[:0],images[:,:,:,:3],images.long(),images*float('nan')):
        with pytest.raises(ValueError):impl.rotation_batch(bad)
    for x,y in ((torch.zeros(2,3),torch.tensor([0,1])),(torch.zeros(2,4),torch.tensor([0,4])),(torch.zeros(2,4),torch.tensor([0.,1.]))):
        with pytest.raises(ValueError):impl.rotation_loss(x,y)


@pytest.mark.t1
def test_exact_ccw_pixels_and_rotation_major_labels(impl):
    """A 2x2 image [[1,2],[3,4]] rotates counterclockwise to [[2,4],[1,3]]."""
    x=torch.tensor([[[[1.,2.],[3.,4.]]],[[[5.,6.],[7.,8.]]]],dtype=torch.float64)
    views,labels=impl.rotation_batch(x)
    expected=[[[1,2],[3,4]],[[5,6],[7,8]],[[2,4],[1,3]],[[6,8],[5,7]],[[4,3],[2,1]],[[8,7],[6,5]],[[3,1],[4,2]],[[7,5],[8,6]]]
    close(views[:,0],torch.tensor(expected,dtype=x.dtype),atol=0,rtol=0)
    assert labels.tolist()==[0,0,1,1,2,2,3,3]


@pytest.mark.t2
def test_scalar_loss_and_reference(impl,ref):
    """Rotation supervision is ordinary four-way mean CE on the self-generated transformation labels."""
    x=torch.tensor([[1.,2.,-.5,0.],[-.3,.2,.8,-1.],[.3,.5,-.9,1.2]],dtype=torch.float64)
    y=torch.tensor([2,0,3]);expected=torch.tensor(scalar_ce(x.tolist(),y.tolist()),dtype=x.dtype)
    close(impl.rotation_loss(x,y),expected,atol=1e-12,rtol=1e-12)
    close(impl.rotation_loss(x,y),ref.rotation_loss(x,y),atol=1e-12,rtol=1e-12)
    close(impl.rotation_loss(torch.zeros_like(x),y),torch.tensor(math.log(4),dtype=x.dtype))


@pytest.mark.t2
def test_rotation_group_and_pixel_multiset(impl):
    """Quarter-turn rotation is a pixel permutation with order four and no interpolation."""
    x=fixture()[:2];z=x
    for _ in range(4):
        views,_=impl.rotation_batch(z);z=views[2:4]
    close(z,x,atol=0,rtol=0)
    views,_=impl.rotation_batch(x)
    for k in range(4):
        for i in range(2):
            assert sorted(views[2*k+i].flatten().tolist())==sorted(x[i].flatten().tolist())


@pytest.mark.t2
def test_rotation_vjp_and_strided_images(impl):
    """Image augmentation keeps the sum of four inverse-permutation VJPs, including strided storage."""
    parent=torch.arange(16,dtype=torch.float64).reshape(1,1,2,8).requires_grad_()
    x=parent[:,:,:,::4]
    views,_=impl.rotation_batch(x);assert views.requires_grad
    weights=torch.arange(16,dtype=torch.float64).reshape(4,1,2,2)/7.
    grad=torch.autograd.grad((views*weights).sum(),parent)[0]
    expected=torch.zeros_like(parent)
    for r in range(2):
        for c in range(2):
            value=weights[0,0,r,c]+weights[1,0,1-c,r]+weights[2,0,1-r,1-c]+weights[3,0,c,1-r]
            expected[0,0,r,4*c]=value
    close(grad,expected,atol=1e-12,rtol=1e-12)
    # Preserve all images/channels while rotating spatial axes only, through multi-axis strides.
    storage=torch.arange(4*6*6*6,dtype=torch.float64).reshape(4,6,6,6).requires_grad_()
    inputs=storage[::2,::2,::2,::2]
    output,labels=impl.rotation_batch(inputs)
    assert output.shape==(8,3,3,3) and labels.tolist()==[0,0,1,1,2,2,3,3]
    weights=torch.arange(8*3*3*3,dtype=storage.dtype).reshape(8,3,3,3)/17.
    expected_values=torch.zeros_like(output);expected_gradient=torch.zeros_like(storage)
    for b in range(2):
        for channel in range(3):
            for row in range(3):
                for column in range(3):
                    destinations=[(row,column),(2-column,row),(2-row,2-column),(column,2-row)]
                    for rotation,(r,c) in enumerate(destinations):
                        expected_values[rotation*2+b,channel,r,c]=inputs.detach()[b,channel,row,column]
                        expected_gradient[2*b,2*channel,2*row,2*column]+=weights[rotation*2+b,channel,r,c]
    close(output,expected_values,atol=0,rtol=0)
    actual_gradient=torch.autograd.grad((output*weights).sum(),storage)[0]
    close(actual_gradient,expected_gradient,atol=1e-12,rtol=1e-12)



@pytest.mark.t2
def test_logit_gradient_and_extreme_offsets(impl):
    """Four-way CE gives (softmax-onehot)/M and must remain finite at large row offsets."""
    x=torch.tensor([[10000.,10002.,9999.,10001.],[-10000.,-9999.,-10002.,-9998.]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,2]);loss=impl.rotation_loss(x,y);expected=[]
    for row,target in zip(x.detach().tolist(),y.tolist()):
        weights=[math.exp(v-max(row)) for v in row]
        expected.append([(v/sum(weights)-(j==target))/2 for j,v in enumerate(weights)])
    close(torch.autograd.grad(loss,x)[0],torch.tensor(expected,dtype=x.dtype),atol=1e-11,rtol=1e-11)
    close(loss,torch.tensor(scalar_ce(x.detach().tolist(),y.tolist()),dtype=x.dtype),atol=1e-11,rtol=1e-11)


@pytest.mark.t2
def test_batch_duplication_label_permutation_and_tiny_images(impl):
    """CE tracks supplied labels and averages examples; 1x1 images still produce all four labels."""
    x=torch.tensor([[1.,2.,0.,-.5],[0.,.2,-.3,1.]],dtype=torch.float64);y=torch.tensor([1,3]);p=[3,1,0,2]
    base=impl.rotation_loss(x,y)
    close(impl.rotation_loss(x[:,p],torch.tensor([p.index(i) for i in y.tolist()])),base,atol=1e-12,rtol=1e-12)
    close(impl.rotation_loss(torch.cat([x,x]),torch.cat([y,y])),base,atol=1e-12,rtol=1e-12)
    views,labels=impl.rotation_batch(torch.tensor([[[[3.]]]],dtype=x.dtype))
    close(views,torch.full((4,1,1,1),3.,dtype=x.dtype),atol=0,rtol=0);assert labels.tolist()==[0,1,2,3]


@pytest.mark.t3
def test_actual_rotation_classifier_training(impl,artifact):
    """The submitted transformations and loss train a real 64-parameter rotation classifier in 80 updates."""
    result=run(impl.rotation_batch,impl.rotation_loss);artifact('rotnet_training',result);m=result['metrics']
    assert m['updates']==80 and m['trainable_parameters']==64 and m['augmented_images']==32
    assert m['final_accuracy']==1. and m['final_accuracy']>m['initial_accuracy']
    assert m['final_loss']<.06 and m['final_loss']<.05*m['initial_loss']
