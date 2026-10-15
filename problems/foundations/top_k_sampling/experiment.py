import math
import torch


def run(solve):
    temperatures=[.25,.5,1.,2.,4.];maxima=[];entropies=[]
    logits=torch.tensor([2.,0.,3.,1.],dtype=torch.float64)
    for temperature in temperatures:
        probs=solve(logits,2,temperature)
        assert int((probs>0).sum())==2 and probs[1]==0 and probs[3]==0
        expected=1/(1+torch.exp(torch.tensor(-1/temperature,dtype=logits.dtype)))
        torch.testing.assert_close(probs[2],expected)
        maxima.append(probs.max().item())
        entropies.append(float(-(probs*probs.clamp(min=1e-30).log()).sum()))
    assert maxima[0]>maxima[-1] and entropies[0]<entropies[-1]
    observed, error = sampled_frequencies(solve(logits, 2, 1.),
                                          [1/(1+math.e), 0., 1/(1+math.exp(-1)), 0.], 42)
    sample_chart = {'kind':'line', 'title':'Actual Top-k sampling (8192 draws)',
                    'x':list(range(4)), 'x_label':'vocabulary index', 'y_label':'frequency',
                    'series':[{'name':'Independent target', 'values':[1/(1+math.e), 0., 1/(1+math.exp(-1)), 0.]},
                              {'name':'Sampled learner distribution', 'values':observed}]}
    return {'provenance':'Synthetic temperature sweep and seeded sampling from learner probabilities; no language-model quality benchmark.',
            'charts':[sample_chart,{'kind':'line','title':'Temperature after Top-k','x':temperatures,'x_label':'temperature','y_label':'probability','series':[{'name':'Largest candidate probability','values':maxima}]},
                      {'kind':'line','title':'Candidate entropy','x':temperatures,'x_label':'temperature','y_label':'entropy (nats)','series':[{'name':'Learner distribution entropy','values':entropies}]}],
            'metrics':{'sample_count':8192,'maximum_sampling_error':error,'candidate_count':2,'cold_entropy':entropies[0],'warm_entropy':entropies[-1]}}


def sampled_frequencies(probs, expected, seed):
    # Per-backend reproducibility; NumPy and native Torch have different RNG streams.
    sample_count = 8192
    torch.manual_seed(seed)
    draws = torch.multinomial(probs, sample_count, replacement=True)
    counts = torch.bincount(draws, minlength=probs.shape[-1])
    observed = counts.to(torch.float64) / sample_count
    target = torch.tensor(expected, dtype=torch.float64)
    torch.testing.assert_close(probs, target)
    assert (counts[target == 0] == 0).all()
    error = float((observed - target).abs().max())
    assert error < .04  # >7 standard errors for these small categorical distributions.
    return observed.tolist(), error
