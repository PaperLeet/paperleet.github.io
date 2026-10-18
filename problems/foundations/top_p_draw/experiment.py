import torch


def run(sample):
    sample_count=8192
    logits=torch.tensor([.1,.5,.15,.25],dtype=torch.float64).log().expand(sample_count,4)
    target=torch.tensor([0.,2/3,0.,1/3],dtype=torch.float64)
    tokens=sample(logits,.6,1.,torch.Generator().manual_seed(42))
    counts=torch.bincount(tokens,minlength=4)
    frequency=counts.to(torch.float64)/sample_count
    assert counts[0]==0 and counts[2]==0, 'Excluded tokens must never be sampled.'
    error=float((frequency-target).abs().max())
    assert error<.04, 'Empirical frequencies must match the nucleus distribution.'
    return {'provenance':'8192 actual learner draws from synthetic logits, seed 42; no language-model quality benchmark.',
            'charts':[{'kind':'line','title':'Tokens sampled by your implementation','x':[0,1,2,3],
                       'x_label':'Vocabulary ID','y_label':'Frequency','series':[
                           {'name':'Independent target','values':target.tolist()},
                           {'name':'Your sampled tokens','values':frequency.tolist()}]}],
            'metrics':{'sample_count':sample_count,'maximum_frequency_error':error}}
