from problems.self_evolving.mmzero_proposer import solution
from problems.self_evolving.mmzero_proposer.experiment import run
import yaml
from pathlib import Path

def generate():
    result=run(solution)
    meta=yaml.safe_load((Path(__file__).parents[1]/'problem.yaml').read_text())
    return {'kind':'toy_experiment','title':meta['title_en'],'problem':meta['id'],
            'provenance':meta['toy'],**result}
