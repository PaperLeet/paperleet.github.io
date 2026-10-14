"""Unconstrained generation followed by exact-ID filtering, not trie decoding."""
import math
from problems.retrieval.generative_retrieval.toy import catalog, next_logits, exhaustive


def run(beam_search, lookup_candidates):
    ids = catalog()
    mapping = {tuple(row):i for i,row in enumerate(ids.tolist())}
    exact = [(mapping[tokens],score) for tokens,score in exhaustive() if tokens in mapping][:3]
    target = {index for index,_ in exact}
    widths, overlaps, valid_counts, invalid_counts = [1,2,4,8,16,625], [], [], []
    maximum_score_error = 0.
    truth = dict(exhaustive())
    for width in widths:
        beams = beam_search(next_logits, 4, width)
        valid = lookup_candidates(beams, ids, 3)
        overlaps.append(len(target & {index for index,_ in valid}) / 3)
        valid_counts.append(len(valid))
        invalid_counts.append(sum(tokens not in mapping for tokens,_ in beams))
        for tokens,score in beams:
            maximum_score_error = max(maximum_score_error, abs(score-truth[tokens]))
    return {'provenance':'Fixed synthetic conditional model, 5 tokens / 4 positions / 8 catalog items; exact enumeration of 625 IDs is a model-ranking oracle, not user relevance or paper Recall/NDCG.',
            'charts':[{'kind':'line','title':'Beam width and valid retrieval','x':[math.log2(w) for w in widths],
                       'x_label':'log2(beam width); widths 1,2,4,8,16,625','y_label':'fraction / count',
                       'series':[{'name':'overlap with exact model top-3','values':overlaps},
                                 {'name':'returned valid items (up to 3)','values':valid_counts}]},
                      {'kind':'line','title':'Invalid IDs before catalog filtering','x':[math.log2(w) for w in widths],
                       'x_label':'log2(beam width)','y_label':'invalid complete IDs',
                       'series':[{'name':'discarded IDs','values':invalid_counts}]}],
            'metrics':{'maximum_log_score_error':maximum_score_error,'exhaustive_top3_overlap':overlaps[-1],
                       'exhaustive_valid_count':valid_counts[-1], 'greedy_top3_overlap':overlaps[0],
                       'full_space_invalid_ids':invalid_counts[-1],
                       'exact_top_item':exact[0][0]}}
