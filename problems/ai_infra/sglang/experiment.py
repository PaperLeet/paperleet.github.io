def evaluate(radix_cache_step):
    prompts=[(1,2,3,4),(1,2,3,5),(1,2,6),(8,9,10),(1,2,3,4),(8,9,11)]
    xs=list(range(1,7)); curves=[]; maximum=0; total=sum(map(len,prompts)); largest=0
    for cap in (7,12):
        cache={}; computed=0; curve=[]
        for clock,p in enumerate(prompts,1):
            cache,hit=radix_cache_step(cache,p,cap,clock); computed+=len(p)-hit; curve.append(computed); maximum=max(maximum,len(cache))
            assert len(cache)<=cap
        curves.append((f'Capacity {cap}',curve))
        if cap==12: largest=computed
    baseline=[]; count=0
    for p in prompts: count+=len(p); baseline.append(count)
    curves.append(('No cache',baseline))
    return {'cached_prefill_tokens':largest,'uncached_prefill_tokens':total,'reused_tokens':total-largest,'maximum_nodes':maximum}, [('Cumulative prefill token work',xs,curves)]


def run(radix_cache_step):
    metrics, charts = evaluate(radix_cache_step)
    return {'kind': 'toy_experiment', 'title': 'SGLang Prefix KV Cache and Leaf-Only LRU', 'problem': 'ai_infra/sglang', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2312.07104', 'anchor': 'Section 3: RadixAttention, longest-prefix reuse, leaf-first LRU and reference counters', 'adaptation': 'Replay six prompts sharing system prefixes with capacities seven and twelve; compare prefill token counts with no caching and verify node capacity.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
