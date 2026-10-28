def radix_cache_step(cache: dict, prompt: tuple[int,...], capacity: int, clock: int, pinned: tuple = ()) -> tuple[dict,int]:
    def valid_path(p):
        return isinstance(p,tuple) and len(p)>0 and all(type(t) is int and t>=0 for t in p)
    if type(capacity) is not int or capacity<=0 or type(clock) is not int or clock<0 or not valid_path(prompt):
        raise ValueError('valid prompt, clock and positive capacity required')
    if len(cache)>capacity or len(prompt)>capacity:
        raise ValueError('cache or prompt exceeds capacity')
    for p,t in cache.items():
        if not valid_path(p) or type(t) is not int or not 0<=t<=clock or any(p[:j] not in cache for j in range(1,len(p))):
            raise ValueError('cache must be prefix closed with valid timestamps')
    if any(p not in cache for p in pinned): raise ValueError('pinned nodes must exist')
    protected={prompt[:j] for j in range(1,len(prompt)+1)}
    for p in pinned:
        protected.update(p[:j] for j in range(1,len(p)+1))
    reused=0
    for j in range(1,len(prompt)+1):
        if prompt[:j] not in cache: break
        reused=j
    result=dict(cache)
    for j in range(1,len(prompt)+1):
        p=prompt[:j]
        if p not in result:
            while len(result)>=capacity:
                parents={node[:-1] for node in result}
                candidates=[node for node in result if node not in parents and node not in protected]
                if not candidates: raise ValueError('capacity exhausted by pinned prefixes')
                victim=min(candidates,key=lambda node:(result[node],node))
                del result[victim]
        result[p]=clock
    return result,reused
