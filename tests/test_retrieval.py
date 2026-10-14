"""Cross-lesson catalog continuity and independent learner implementation acceptance."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from paperleet.catalog import problem, root
from paperleet.browser_judge import export_browser_bundle
from paperleet.judge.runner import run_problem
from problems.retrieval.semantic_ids.toy import data
from problems.retrieval.generative_retrieval.toy import catalog
from problems.retrieval.semantic_ids.solution import residual_quantize, resolve_collisions


def test_quantized_items_are_the_actual_retrieval_catalog():
    x,books=data()
    codes,_=residual_quantize(x,books)
    assert resolve_collisions(codes).tolist()==catalog().tolist()


ALTERNATIVES={
 'semantic_ids': '''import torch
from torch import Tensor

@torch.no_grad()
def residual_quantize(latents: Tensor, codebooks: Tensor):
    rows=[];values=[]
    for row in latents.tolist():
        remaining=list(row);reconstruction=[0.]*len(row);indices=[]
        for book in codebooks.tolist():
            distances=[sum((a-b)**2 for a,b in zip(remaining,entry)) for entry in book]
            index=distances.index(min(distances));indices.append(index)
            remaining=[a-b for a,b in zip(remaining,book[index])]
            reconstruction=[a+b for a,b in zip(reconstruction,book[index])]
        rows.append(indices);values.append(reconstruction)
    return (torch.tensor(rows,dtype=torch.long).reshape(latents.shape[0],codebooks.shape[0]),
            torch.tensor(values,dtype=latents.dtype).reshape(latents.shape))

def resolve_collisions(codes: Tensor):
    rows=codes.tolist()
    suffixes=[sum(previous==row for previous in rows[:i]) for i,row in enumerate(rows)]
    return torch.cat([codes,torch.tensor(suffixes,dtype=torch.long).reshape(-1,1)],dim=1)
''',
 'generative_retrieval': '''import math

def beam_search(next_logits, length, beam_width):
    current=[((),0.)]
    for _ in range(length):
        expanded=[]
        for prefix,score in current:
            values=next_logits(prefix).tolist();peak=max(values)
            log_sum=math.log(sum(math.exp(value-peak) for value in values))
            expanded.extend((prefix+(i,),score+(value-peak)-log_sum) for i,value in enumerate(values))
        current=sorted(expanded,key=lambda item:(-item[1],item[0]))[:beam_width]
    return current

def lookup_candidates(beams,catalog_ids,k):
    results=[];rows=catalog_ids.tolist()
    for tokens,score in beams:
        if len(results)>=k:break
        matches=[i for i,row in enumerate(rows) if tuple(row)==tokens]
        if matches and not any(i==matches[0] for i,_ in results):results.append((matches[0],score))
    return results
'''}


@pytest.mark.parametrize('slug',ALTERNATIVES)
def test_independent_scalar_learner_passes_native_and_browser_suite(tmp_path,slug):
    identifier='retrieval/'+slug
    path=tmp_path/'learner.py';path.write_text(ALTERNATIVES[slug])
    native=run_problem(identifier,path)
    assert native['passed'],native
    output=tmp_path/'bundle'
    metadata=export_browser_bundle(problem(identifier),output)
    script='''import json,sys
from pathlib import Path
from paperleet.browser.harness import prepare,run_suite
output=Path(sys.argv[1]);metadata=json.loads(sys.argv[2])
bundle=json.loads((output/metadata['bundle_url']).read_text())
runtime=json.loads((output/bundle['runtime_bundle_url']).read_text())
bundle={**bundle,'adapter':runtime['adapter'],'harness':runtime['harness'],'files':{**runtime['files'],**bundle['files']}}
prepare(bundle,bundle['adapter'],sys.argv[3])
report=run_suite(bundle,Path(sys.argv[4]).read_text(),sys.argv[3])
assert report['passed'],report
print(json.dumps({'passed':report['passed'],'tests':len(report['tests'])}))
'''
    result=subprocess.run([sys.executable,'-c',script,str(output),json.dumps(metadata),str(tmp_path/'judge'),str(path)],cwd=root(),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)=={'passed':True,'tests':8}
