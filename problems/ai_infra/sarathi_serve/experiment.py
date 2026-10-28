def evaluate(hybrid_batch):
    pending=[13,3,9]; prefill_work=0; stalls=0; counts=[]; chunks=[]; remaining=[]
    for _ in range(30):
        if not pending: break
        r=hybrid_batch([100,101],pending,8,4)
        stalls+=int(r['decodes']!=[100,101]); work=sum(n for _,n in r['prefills']); prefill_work+=work
        counts.append(len(r['decodes'])+work); chunks.extend(n for _,n in r['prefills']); pending=[n for n in r['remaining'] if n>0]; remaining.append(sum(pending))
    assert not pending
    return {'prefill_tokens':prefill_work,'decode_stalls':stalls,'maximum_batch_tokens':max(counts),'maximum_chunk':max(chunks)}, [('Hybrid token budget',list(range(1,len(counts)+1)),[('Batch tokens',counts),('Remaining prefill tokens',remaining)])]


def run(hybrid_batch):
    metrics, charts = evaluate(hybrid_batch)
    return {'kind': 'toy_experiment', 'title': 'Sarathi-Serve Decode Priority and Chunked Prefill', 'problem': 'ai_infra/sarathi_serve', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2403.02310', 'anchor': 'Section 4.1–4.2; Algorithm 3, stall-free batching', 'adaptation': 'Keep two decodes active with prefill lengths 13, 3 and 9, budget eight and chunk cap four. Consume the queue over multiple iterations and check uninterrupted decodes and conserved prefill work.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
