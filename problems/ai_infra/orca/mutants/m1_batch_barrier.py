# Bug: 必须旧批全部结束才接纳请求
def iteration_schedule(arrivals: list[int], lengths: list[int], capacity: int) -> tuple[list[tuple],list[int]]:
    if len(arrivals)!=len(lengths) or type(capacity) is not int or capacity<=0:
        raise ValueError('matching requests and positive capacity required')
    if any(type(a) is not int or a<0 for a in arrivals) or any(type(n) is not int or n<=0 for n in lengths):
        raise ValueError('nonnegative integer arrivals and positive lengths required')
    pending=sorted(range(len(arrivals)),key=lambda i:(arrivals[i],i)); active=[]; remaining=list(lengths); finished=[-1]*len(lengths); batches=[]; t=0
    while pending or active:
        while pending and arrivals[pending[0]]<=t and len(active)<capacity and (not active or t==0):
            active.append(pending.pop(0))
        batches.append(tuple(active))
        for i in active:
            remaining[i]-=1
            if remaining[i]==0: finished[i]=t+1
        active=[i for i in active if remaining[i]>0]
        t+=1
    return batches,finished
