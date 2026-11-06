"""Independent scalar teacher probabilities and cross-view objective."""
import math
import torch


def probabilities(rows,center,temperature):
    result=[]
    for view in rows:
        output=[]
        for row in view:
            logits=[(v-c)/temperature for v,c in zip(row,center)];peak=max(logits)
            weights=[math.exp(v-peak) for v in logits];mass=sum(weights)
            output.append([w/mass for w in weights])
        result.append(output)
    return result


def scalar_loss(student,teacher,center,st=.1,tt=.04):
    q=probabilities(teacher,center,tt);terms=[]
    for t in range(2):
        for v in range(len(student)):
            if t==v:continue
            subtotal=0.
            for row,prob in zip(student[v],q[t]):
                scores=[x/st for x in row];peak=max(scores)
                logsum=peak+math.log(sum(math.exp(x-peak) for x in scores))
                subtotal+=sum(p*(logsum-x) for p,x in zip(prob,scores))
            terms.append(subtotal/len(student[v]))
    return sum(terms)/len(terms)


def fixture(dtype=torch.float64):
    s=torch.arange(36,dtype=dtype).reshape(3,4,3)
    t=torch.arange(24,dtype=dtype).reshape(2,4,3)
    return (s*.37).sin()*.4,(t*.23+.3).cos()*.3,torch.tensor([.1,-.08,.03],dtype=dtype)
