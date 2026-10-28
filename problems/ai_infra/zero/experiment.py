import torch

def evaluate(zero2_adam_step,state_bytes):
    x=torch.cos(torch.arange(48,dtype=torch.float64).reshape(12,4)*.37); target=x@torch.tensor([.6,-.7,.3,1.],dtype=torch.float64)
    p=torch.zeros(2,2,dtype=torch.float64); m=p.clone(); v=p.clone(); dense=p.reshape(-1).clone(); dm=dense.clone(); dv=dense.clone(); losses=[]; errors=[]
    for step in range(1,46):
        gradients=torch.stack([2*x[r*6:(r+1)*6].T@(x[r*6:(r+1)*6]@p.reshape(-1)-target[r*6:(r+1)*6])/6 for r in range(2)])
        p,m,v=zero2_adam_step(p,gradients,m,v,step,.04)
        g=2*x.T@(x@dense-target)/12; dm=.9*dm+.1*g; dv=.999*dv+.001*g*g
        dense=dense-.04*(dm/(1-.9**step))/(torch.sqrt(dv/(1-.999**step))+1e-8)
        losses.append(float(((x@p.reshape(-1)-target)**2).mean())); errors.append(float((p.reshape(-1)-dense).abs().max()))
    memory=[state_bytes(1024,8,s) for s in range(4)]
    return {'initial_loss':losses[0],'final_loss':losses[-1],'trajectory_error':max(errors),'replicated_bytes':memory[0],'stage2_bytes':memory[2]}, [('Sharded regression',list(range(1,46)),[('MSE',losses)]),('Model-state bytes per rank',list(range(4)),[('Bytes',memory)])]


def run(zero2_adam_step, state_bytes):
    metrics, charts = evaluate(zero2_adam_step, state_bytes)
    return {'kind': 'toy_experiment', 'title': 'ZeRO-2 Sharded Adam and State Memory', 'problem': 'ai_infra/zero', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/1910.02054', 'anchor': 'Sections 5.1–5.3: optimizer, gradient and parameter partitioning', 'adaptation': 'Two ranks process deterministic regression examples for 45 Adam steps at learning rate 0.04. Compare with an independent unsharded Adam update and report losses and model-state bytes.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
