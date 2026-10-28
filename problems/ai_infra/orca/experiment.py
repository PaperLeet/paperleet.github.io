def evaluate(iteration_schedule):
    lengths=[6,1,1,1,1,1]; capacities=[1,2,3]; dynamic=[]; static=[]; tokens=0
    for c in capacities:
        batches,ends=iteration_schedule([0]*6,lengths,c); dynamic.append(sum(ends)/6)
        clock=0; completed=[]
        for start in range(0,6,c):
            group=lengths[start:start+c]; clock+=max(group); completed.extend([clock]*len(group))
        static.append(sum(completed)/6)
        if c==2: tokens=sum(len(batch) for batch in batches)
    return {'dynamic_mean_latency':dynamic[1],'static_mean_latency':static[1],'tokens_executed':tokens}, [('Mean completion latency',capacities,[('Iteration admission',dynamic),('Fixed request batches',static)])]


def run(iteration_schedule):
    metrics, charts = evaluate(iteration_schedule)
    return {'kind': 'toy_experiment', 'title': 'Orca Iteration-Level Dynamic Batching', 'problem': 'ai_infra/orca', 'provenance': {'kind': 'teaching_adaptation', 'source': 'https://www.usenix.org/conference/osdi22/presentation/yu', 'anchor': 'Section 3, S1 and Figure 4: iteration-level scheduling; Section 4.2, Algorithm 1: FCFS admission', 'adaptation': 'Replay six simultaneous requests of unequal length, compare completions with request-level fixed batches, and sweep capacities 1, 2 and 3 for latency and slot occupancy.'}, 'metrics': metrics, 'charts': [{'kind':'line','title': title,'x_label':'Experiment coordinate','x':x,'series':[{'name':name,'values':values} for name,values in series]} for title,x,series in charts]}
