# Bug: prefill误用decode保留的预算
def hybrid_batch(decodes: list[int], prefills: list[int], budget: int, chunk_size: int) -> dict:
    if type(budget) is not int or type(chunk_size) is not int or budget<0 or chunk_size<=0 or len(decodes)>budget:
        raise ValueError('budget must fit ongoing decodes and chunk size must be positive')
    if len(set(decodes))!=len(decodes) or any(type(i) is not int or i<0 for i in decodes) or any(type(n) is not int or n<=0 for n in prefills):
        raise ValueError('unique decode IDs and positive prefill lengths required')
    remaining=list(prefills); scheduled=[]; left=budget
    for i,n in enumerate(prefills):
        if left==0: break
        count=min(n,chunk_size,left)
        scheduled.append((i,count)); remaining[i]-=count; left-=count
        if remaining[i]>0: break
    return {'decodes':list(decodes),'prefills':scheduled,'remaining':remaining}
