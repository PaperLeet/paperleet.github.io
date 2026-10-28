"""Deterministic CPU teaching experiment; no model or data downloads."""
import torch
TITLE='DAgger'
IDENTIFIER='robotics/dagger'
SOURCE='https://proceedings.mlr.press/v15/ross11a.html'
ANCHOR='Algorithm 3.1: mixed roll-in policy and aggregation of expert-labeled visited states'
ADAPTATION='Isolate Algorithm 3.1 in a scalar linear environment with policy refitting; no interactive human expert or full classifier.'

def _report(metrics, values, label):
    return {"kind":"toy_experiment", "title":TITLE, "problem":IDENTIFIER,
      "provenance":{"kind":"teaching_adaptation", "source":SOURCE, "anchor":ANCHOR, "adaptation":ADAPTATION},
      "metrics":metrics,"charts":[{"kind":"line","title":label,"x_label":"Step / example", "x":list(range(len(values))),"series":[{"name":label,"values":values}]}]}

def run(mixture_actions,aggregate_dataset):
    start=torch.linspace(-1.,1.,8,dtype=torch.float64).reshape(8,1)
    x=torch.zeros(2,1,dtype=start.dtype); y=x.clone(); gain=0.; curve=[]
    def energy(k):
        state=start; total=0.
        for _ in range(12): state=1.1*state+k*state;total+=state.square().mean().item()
        return total
    initial=energy(gain)
    for iteration in range(4):
        state=start; visited=[]; labels=[]
        for t in range(8):
            visited.append(state);labels.append(-.6*state)
            draws=torch.arange(8,dtype=state.dtype)/8
            action=mixture_actions(-.6*state,gain*state,draws,.5/(iteration+1))
            state=1.1*state+action
        x,y=aggregate_dataset(x,y,torch.cat(visited,0),torch.cat(labels,0))
        gain=((x*y).sum()/x.square().sum()).item();curve.append(energy(gain))
    return _report({'initial_energy':initial,'final_energy':curve[-1],'examples':x.shape[0],'gain':gain},curve,'Closed-loop state energy')
