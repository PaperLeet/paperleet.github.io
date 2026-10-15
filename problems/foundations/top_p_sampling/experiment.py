from paperleet.common.asserts import assert_close_diagnostic as close
import torch


def run(solve):
    thresholds=[.1,.4,.6,.85,1.];counts=[];maxima=[]
    logits=torch.tensor([.1,.5,.15,.25],dtype=torch.float64).log()
    for p in thresholds:
        probs=solve(logits,p)
        close(probs.sum(),torch.tensor(1.,dtype=logits.dtype))
        counts.append(int((probs>0).sum()));maxima.append(probs.max().item())
    assert counts==[1,1,2,3,4]
    close(solve(logits,.6),torch.tensor([0.,2/3,0.,1/3],dtype=logits.dtype))
    observed, error = sampled_frequencies(solve(logits, .6), [0., 2/3, 0., 1/3], 42)
    sample_chart = {'kind':'line', 'title':'Actual nucleus sampling (8192 draws)',
                    'x':list(range(4)), 'x_label':'vocabulary index', 'y_label':'frequency',
                    'series':[{'name':'Independent target', 'values':[0., 2/3, 0., 1/3]},
                              {'name':'Sampled learner distribution', 'values':observed}]}
    return {'provenance':'Synthetic nucleus selection and seeded sampling from learner probabilities; no language-model quality benchmark.',
            'charts':[sample_chart,{'kind':'line','title':'Nucleus size','x':thresholds,'x_label':'p threshold','y_label':'retained candidates','series':[{'name':'Learner candidate count','values':counts}]},
                      {'kind':'line','title':'Renormalized candidate confidence','x':thresholds,'x_label':'p threshold','y_label':'largest probability','series':[{'name':'Learner maximum probability','values':maxima}]}],
            'metrics':{'sample_count':8192,'maximum_sampling_error':error,'small_nucleus':counts[0],'full_nucleus':counts[-1]}}


def sampled_frequencies(probs, expected, seed):
    # Per-backend reproducibility; NumPy and native Torch have different RNG streams.
    sample_count = 8192
    torch.manual_seed(seed)
    draws = torch.multinomial(probs, sample_count, replacement=True)
    counts = torch.bincount(draws, minlength=probs.shape[-1])
    observed = counts.to(torch.float64) / sample_count
    target = torch.tensor(expected, dtype=torch.float64)
    close(probs, target)
    assert (counts[target == 0] == 0).all()
    error = float((observed - target).abs().max())
    assert error < .04  # >7 standard errors for these small categorical distributions.
    return observed.tolist(), error
