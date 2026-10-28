def evaluate(disaggregated_trace,slo_goodput):
    transfer=[0,1,2,4,8]; goodputs=[]; ends=[]; final_first=0
    for delay in transfer:
        first,last=disaggregated_trace([0,0,0],[4,4,4],[1,1,1],[delay]*3,[3,3,3])
        goodputs.append(slo_goodput([0,0,0],first,last,[3,3,3],12,2)); ends.append(last[-1]); final_first=first[-1]
    return {'first_token_last_request':final_first,'zero_transfer_goodput':goodputs[0],'long_transfer_goodput':goodputs[-1]}, [('Transfer and compliant throughput',transfer,[('Goodput',goodputs)]),('Last token time',transfer,[('Completion',ends)])]


def run(disaggregated_trace, slo_goodput):
    metrics, charts = evaluate(disaggregated_trace, slo_goodput)
    return {'kind': 'toy_experiment', 'title': 'DistServe Disaggregated Queues and Dual Latency SLOs', 'problem': 'ai_infra/distserve', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin', 'anchor': 'Sections 2 and 3: TTFT/TPOT, disaggregated prefill and decoding, goodput', 'adaptation': 'Replay three simultaneous requests with prefill cost four, decode cost one and three output tokens, sweeping KV-transfer time. Report first/last-token times and jointly compliant goodput.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
