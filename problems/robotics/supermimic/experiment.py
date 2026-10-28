"""A fixed differentiable 1D world model trains a shared goal-conditioned policy."""
import torch


def _imagine(gain):
    # These fixed dynamics are a stand-in for a fitted and temporarily frozen model.
    state = torch.tensor([-.8,-.4,.1,.6],dtype=gain.dtype)
    goals = torch.ones_like(state)
    rewards,termination,values,errors = [],[],[],[]
    for step in range(8):
        values.append(2*torch.exp(-2*(state-goals).square()))
        action = gain*(goals-state)
        state = .9*state + .25*action
        error = (state-goals).square()
        rewards.append(torch.exp(-2*error)-.01*action.square())
        termination.append(.05*torch.sigmoid(4*error-3))
        errors.append(error)
    values.append(2*torch.exp(-2*(state-goals).square()))
    return torch.stack(rewards,1),torch.stack(termination,1),torch.stack(values,1),torch.stack(errors,1)


def run(imagined_return, lambda_return_targets):
    gain = torch.tensor(.15,dtype=torch.float64,requires_grad=True)
    returns,tracking,gains = [],[],[]
    for step in range(101):
        rewards,done,values,errors = _imagine(gain)
        objective = imagined_return(rewards,done,values[:,-1],.95)
        returns.append(objective.item()); tracking.append(errors.mean().item()); gains.append(gain.item())
        if step < 100:
            gradient, = torch.autograd.grad(objective,gain)
            gain = (gain+.08*gradient).detach().requires_grad_()
    targets = lambda_return_targets(rewards,done,values,.95,1.)
    return {'kind':'toy_experiment','title':'SuperMimic: policy gradients through a fixed world model',
        'problem':'robotics/supermimic',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2610.28480',
          'anchor':'Section 3.3, Eq. (2) soft-survival actor return and Eq. (3) critic targets',
          'adaptation':'A single feedback gain controls four 1D states for eight imagined steps through fixed analytic dynamics and analytic heads. This isolates return gradients, not humanoid tracking or world-model fitting.'},
        'metrics':{'initial_return':returns[0],'final_return':returns[-1],
          'initial_tracking_error':tracking[0],'final_tracking_error':tracking[-1],
          'lambda_one_consistency_error':abs(targets[:,0].mean().item()-returns[-1])},
        'charts':[{'kind':'line','title':'Imagined return','x_label':'Policy update','x':list(range(len(returns))),
          'series':[{'name':'Bootstrapped return','values':returns}]},
          {'kind':'line','title':'Tracking and policy gain','x_label':'Policy update','x':list(range(len(returns))),
           'series':[{'name':'Mean tracking error','values':tracking},{'name':'Feedback gain','values':gains}]}]}
