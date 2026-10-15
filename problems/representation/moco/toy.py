"""Independent scalar dictionary/EMA/loss oracles and a fixed tiny fixture."""
import math
import torch


def normalized(rows):
    return [[v / max(math.sqrt(sum(x*x for x in row)),1e-12) for v in row] for row in rows]


def scalar_loss(query, key, queue, temperature):
    q,k,negative=map(normalized,(query,key,queue));losses=[]
    for a,b in zip(q,k):
        logits=[sum(x*y for x,y in zip(a,b))/temperature]
        logits += [sum(x*y for x,y in zip(a,row))/temperature for row in negative]
        maximum=max(logits)
        losses.append(maximum+math.log(sum(math.exp(v-maximum) for v in logits))-logits[0])
    return sum(losses)/len(losses)


def matmul(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def scalar_step(query, key, queue, pointer, xq, xk, temperature, lr, momentum):
    q,k=matmul(xq,query),matmul(xk,key)
    loss=scalar_loss(q,k,queue,temperature);gradient=[];h=1e-6
    for i,row in enumerate(query):
        gradient.append([])
        for j,value in enumerate(row):
            plus=[r[:] for r in query];minus=[r[:] for r in query]
            plus[i][j]=value+h;minus[i][j]=value-h
            gradient[-1].append((scalar_loss(matmul(xq,plus),k,queue,temperature)-scalar_loss(matmul(xq,minus),k,queue,temperature))/(2*h))
    next_query=[[v-lr*g for v,g in zip(row,grad)] for row,grad in zip(query,gradient)]
    next_key=[[momentum*b+(1-momentum)*a for a,b in zip(row,old)] for row,old in zip(next_query,key)]
    next_queue=[row[:] for row in queue]
    for i,row in enumerate(normalized(k)):next_queue[(pointer+i)%len(queue)]=row
    return loss,next_query,next_key,next_queue,(pointer+len(k))%len(queue)


def fixture():
    features=[[math.sin((i+1)*(j+2)*.7)+.2*math.cos(i+j) for j in range(4)] for i in range(6)]
    other=[[v+.15*math.cos((i+2)*(j+1)) for j,v in enumerate(row)] for i,row in enumerate(features)]
    weights=[[math.cos((i+1)*(j+1)*.4)/2 for j in range(3)] for i in range(4)]
    queue=normalized([[math.sin((i+2)*(j+1)*.6) for j in range(3)] for i in range(5)])
    return tuple(torch.tensor(v,dtype=torch.float64) for v in (features,other,weights,queue))
