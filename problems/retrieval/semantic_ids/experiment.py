"""Frozen residual quantization diagnostics from the supplied learner functions."""
import math
from problems.retrieval.semantic_ids.toy import data, scalar_quantize


def run(residual_quantize, resolve_collisions):
    latents, books = data()
    depths, errors, oracle_error = [1,2,3], [], 0.
    codes = None
    for depth in depths:
        codes, reconstruction = residual_quantize(latents, books[:depth])
        expected_codes, expected_values = scalar_quantize(latents.tolist(), books[:depth].tolist())
        oracle_error = max(oracle_error, max(abs(a-b) for row, exp in zip(reconstruction.tolist(),expected_values) for a,b in zip(row,exp)))
        assert codes.tolist() == expected_codes, 'Selected residual code differs from scalar oracle'
        errors.append(math.sqrt((latents-reconstruction).square().mean().item()))
    ids = resolve_collisions(codes)
    expected_suffixes = [0,1,2,0,0,0,0,0]
    suffix_error = max(abs(a-b) for a,b in zip(ids[:,-1].tolist(),expected_suffixes))
    return {'provenance': 'Synthetic frozen TIGER quantizer, 8 items / 3 levels / 5 codes / 2 features; no RQ-VAE training or paper benchmark.',
            'charts': [
              {'kind':'line','title':'Remaining quantization error','x':depths,'x_label':'residual levels','y_label':'latent RMSE',
               'series':[{'name':'learner reconstruction','values':errors}]},
              {'kind':'line','title':'Collision suffix by catalog row','x':list(range(8)),'x_label':'item row','y_label':'collision ordinal',
               'series':[{'name':'learner suffix','values':ids[:,-1].tolist()}]}],
            'metrics': {'maximum_scalar_error':oracle_error,'suffix_error':suffix_error,
                        'unique_full_ids':len(set(map(tuple,ids.tolist()))),
                        'semantic_prefixes':len(set(map(tuple,codes.tolist()))),
                        'initial_rmse':errors[0],'final_rmse':errors[-1]}}
