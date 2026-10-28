# Bug: 让prefill错误等待decode完成
import math

def disaggregated_trace(arrivals: list[float], prefill: list[float], decode: list[float], transfer: list[float], tokens: list[int]) -> tuple[list[float],list[float]]:
    n=len(arrivals)
    if any(len(v)!=n for v in (prefill,decode,transfer,tokens)) or arrivals!=sorted(arrivals):
        raise ValueError('matching FCFS request lists required')
    if any(not math.isfinite(x) or x<0 for x in arrivals+transfer) or any(not math.isfinite(x) or x<=0 for x in prefill+decode) or any(type(t) is not int or t<2 for t in tokens):
        raise ValueError('finite valid costs and at least two output tokens required')
    pclock=0.; dclock=0.; first=[]; last=[]
    for a,p,d,k,nout in zip(arrivals,prefill,decode,transfer,tokens):
        pclock=max(pclock,dclock,a)+p
        dclock=max(dclock,pclock+k)+(nout-1)*d
        first.append(pclock); last.append(dclock)
    return first,last

def slo_goodput(arrivals: list[float], first: list[float], last: list[float], tokens: list[int], ttft_limit: float, tpot_limit: float) -> float:
    if not len(arrivals)==len(first)==len(last)==len(tokens) or not math.isfinite(ttft_limit) or not math.isfinite(tpot_limit) or min(ttft_limit,tpot_limit)<0:
        raise ValueError('matching traces and finite nonnegative limits required')
    if not arrivals: return 0.
    if any(type(n) is not int or n<2 or not all(math.isfinite(v) for v in (a,f,e)) or a<0 or f<a or e<=f for a,f,e,n in zip(arrivals,first,last,tokens)):
        raise ValueError('invalid completion trace')
    successes=sum(f-a<=ttft_limit and (e-f)/(n-1)<=tpot_limit for a,f,e,n in zip(arrivals,first,last,tokens))
    return successes/(max(last)-min(arrivals))
