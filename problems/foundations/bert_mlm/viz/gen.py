from problems.foundations.bert_mlm.solution import make_mlm_inputs, masked_lm_loss
from problems.foundations.bert_mlm.experiment import run


def generate():
    result = run(make_mlm_inputs, masked_lm_loss)
    return {'kind': 'toy_experiment', 'title': 'BERT: Corruption and Masked Language Modeling', 'problem': 'foundations/bert_mlm',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/1810.04805v2#S3.SS1',
                           'anchor': 'Section 3.1, Task #1: Masked LM; Appendix C.2: masking procedures', 'adaptation': 'The supplied 64 three-token sequences cover 16 ordered left/right pairs; the center is `(2*left+right)%4`. Only centers are eligible, selected at rate 15%. Your corruption function supplies inputs/labels. A 16×4 conditional table reads only the corrupted left/right tokens and trains through your MLM loss for 100 Adam updates (lr=0.08, seed=163, float64). The chart evaluates all centers. Acceptance requires final NLL<0.1 and <10% of initial, with context accuracy>=95%. This is a synthetic conditional-table experiment, without a Transformer or natural-language generalization claim.'},
            'charts': result['charts'], 'metrics': result['metrics']}
