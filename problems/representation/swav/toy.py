"""Independent scalar Sinkhorn and cross-entropy for modest test scores."""
import math
import torch


def scalar_assignments(rows,epsilon=.05,iterations=3):
    b,k=len(rows),len(rows[0]);peak=max(v/epsilon for row in rows for v in row)
    q=[[math.exp(rows[i][j]/epsilon-peak) for i in range(b)] for j in range(k)]
    mass=sum(sum(row) for row in q);q=[[v/mass for v in row] for row in q]
    for _ in range(iterations):
        q=[[v/sum(row)/k for v in row] for row in q]
        columns=[sum(q[j][i] for j in range(k)) for i in range(b)]
        q=[[q[j][i]/columns[i]/b for i in range(b)] for j in range(k)]
    return [[q[j][i]*b for j in range(k)] for i in range(b)]


def scalar_loss(a,b,q1,q2,temperature=.1):
    total=0.
    for scores,assignment in ((a,q2),(b,q1)):
        for row,prob in zip(scores,assignment):
            values=[v/temperature for v in row];peak=max(values)
            logsum=peak+math.log(sum(math.exp(v-peak) for v in values))
            total+=sum(q*(logsum-s) for q,s in zip(prob,values))
    return total/(2*len(a))


def fixture(dtype=torch.float64):
    grid=torch.arange(18,dtype=dtype).reshape(6,3)
    return (grid*.47).sin()*.3,(grid*.31+.2).cos()*.4
