"""Deterministic action-space teaching experiment with a two-parameter denoiser."""
import torch


def run(add_action_noise, action_denoising_loss, action_chunk):
    dtype = torch.float64
    goals = torch.linspace(-1.2,1.2,32,dtype=dtype).reshape(32,1,1)
    profile = torch.tensor([.2,.35,.5,.65,.8],dtype=dtype).reshape(1,5,1)
    axes = torch.tensor([1.,-.7],dtype=dtype).reshape(1,1,2)
    clean = goals * profile * axes
    noise = torch.sin(torch.arange(clean.numel(),dtype=dtype)*1.71).reshape_as(clean)
    alpha = torch.full((32,),.64,dtype=dtype)
    noisy = add_action_noise(clean,noise,alpha)
    # Observation-conditioned basis: a goal and known toy action-time profile.
    condition = goals * profile * axes
    weights = torch.zeros(2,dtype=dtype,requires_grad=True)
    losses = []
    for step in range(301):
        predicted = weights[0]*noisy + weights[1]*condition
        loss = action_denoising_loss(predicted,noise)
        losses.append(loss.item())
        if step < 300:
            gradient, = torch.autograd.grad(loss,weights)
            weights = (weights-.6*gradient).detach().requires_grad_()
    recovered = (noisy-.6*predicted.detach())/.8
    executed = action_chunk(recovered,2,3)
    expected = clean[:,1:4]
    unconditioned = (noisy-.6*(weights[0].detach()*noisy))/.8
    return {
        'kind':'toy_experiment','title':'Diffusion Policy: conditioned action denoising',
        'problem':'robotics/diffusion_policy',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2303.04137',
          'anchor':'Section 2.3 conditional action diffusion; official predict_action observation/action alignment',
          'adaptation':'One fixed noise level and two learned coefficients denoise analytic 2D action sequences conditioned on a scalar goal. No image encoder, multistep sampler or robot benchmark.'},
        'metrics':{'initial_loss':losses[0],'final_loss':losses[-1],
          'executed_chunk_mse':(executed-expected).square().mean().item(),
          'without_conditioning_mse':(unconditioned[:,1:4]-expected).square().mean().item()},
        'charts':[{'kind':'line','title':'Conditional noise prediction loss','x_label':'Gradient step',
          'x':list(range(len(losses))),'series':[{'name':'Noise MSE','values':losses}]},
          {'kind':'line','title':'Planned action sequence for one goal','x_label':'Prediction index',
           'x':list(range(5)),'series':[{'name':'Demonstration','values':clean[-1,:,0].tolist()},
           {'name':'Denoised prediction','values':recovered[-1,:,0].tolist()},
           {'name':'Without goal condition','values':unconditioned[-1,:,0].tolist()}]}]}
