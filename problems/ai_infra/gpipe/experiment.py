import torch

def evaluate(pipeline_schedule,microbatch_mse):
    x=torch.sin(torch.arange(44,dtype=torch.float64).reshape(11,4)*.31); target=x@torch.tensor([.8,-.5,.2,1.1],dtype=torch.float64)
    w=torch.zeros(4,dtype=torch.float64); dense=w.clone(); losses=[]; errors=[]
    for _ in range(35):
        w=w.detach().requires_grad_(); loss=microbatch_mse(x,target,w,4)
        grad,=torch.autograd.grad(loss,(w,)); w=w-.08*grad
        derror=x@dense-target; dense=dense-.08*(2*x.T@derror/11)
        losses.append(float(loss.detach())); errors.append(float((w-dense).abs().max()))
    counts=[1,2,4,8,16]; utilizations=[]
    for m in counts:
        events=pipeline_schedule(4,m); slots=max(e[3] for e in events)+1
        utilizations.append(len(events)/(4*slots))
    return {'initial_loss':losses[0],'final_loss':losses[-1],'trajectory_error':max(errors),'utilization_first':utilizations[0],'utilization_last':utilizations[-1]}, [('Training loss',list(range(35)), [('Microbatch SGD',losses)]),('Pipeline useful-slot fraction',counts,[('Four stages',utilizations)])]


def run(pipeline_schedule, microbatch_mse):
    metrics, charts = evaluate(pipeline_schedule, microbatch_mse)
    return {'kind': 'toy_experiment', 'title': 'GPipe Fill-Drain Scheduling and Microbatch Gradients', 'problem': 'ai_infra/gpipe', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/1811.06965', 'anchor': 'Section 2.2, Algorithm; Figure 2, synchronous microbatch pipeline', 'adaptation': 'Train a four-feature linear model on 11 deterministic samples for 35 SGD steps at learning rate 0.08; compare full-batch and size-four chunks. Also compute useful-slot fractions for four stages as microbatch count varies.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
