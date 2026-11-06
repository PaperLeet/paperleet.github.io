"""Learn three student score tables while updating a center from fixed teacher logits."""
import torch
from .toy import scalar_loss,probabilities


def run(teacher_probabilities,dino_loss,update_center):
    identity=torch.tensor([[1.2,-.4,-.4],[-.4,1.2,-.4],[-.4,-.4,1.2]]*2,dtype=torch.float64)
    bias=torch.tensor([.7,-.3,.2],dtype=identity.dtype)
    teacher=torch.stack((identity+bias,identity*1.1+bias+.02),0).requires_grad_()
    grid=torch.arange(54,dtype=identity.dtype).reshape(3,6,3)
    student=((grid*.29).sin()*.1).requires_grad_();center=torch.zeros(3,dtype=identity.dtype)
    mean_teacher=[sum(teacher.detach().tolist()[v][i][k] for v in range(2) for i in range(6))/12 for k in range(3)]
    optimizer=torch.optim.SGD([student],lr=.6)
    steps,losses,center_norm=[],[],[];scalar_error=0.;center_error=0.;probability_error=0.
    for step in range(81):
        loss=dino_loss(student,teacher,center,.3,.15)
        expected=scalar_loss(student.detach().tolist(),teacher.detach().tolist(),center.tolist(),.3,.15)
        scalar_error=max(scalar_error,abs(float(loss.detach())-expected))
        q=teacher_probabilities(teacher,center,.15)
        oracle_q=probabilities(teacher.detach().tolist(),center.tolist(),.15)
        probability_error=max(probability_error,max(abs(x-y) for a,b in zip(q.tolist(),oracle_q) for ra,rb in zip(a,b) for x,y in zip(ra,rb)))
        if step%8==0:steps.append(step);losses.append(float(loss.detach()));center_norm.append(float(center.norm()))
        if step<80:
            optimizer.zero_grad();loss.backward();optimizer.step()
            center=update_center(center,teacher,.9)
            expected_center=[x*(1-.9**(step+1)) for x in mean_teacher]
            center_error=max(center_error,max(abs(x-y) for x,y in zip(center.tolist(),expected_center)))
    predictions=student.detach().argmax(-1).tolist()
    accuracy=sum(int(value==i%3) for view in predictions for i,value in enumerate(view))/18
    return {'kind':'toy_experiment','title':'DINO: sharpened targets and moving center','problem':'representation/dino',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2104.14294','anchor':'Section 3 and official DINOLoss.forward/update_center',
            'adaptation':'80 actual SGD updates on three 6x3 student score tables against two fixed teacher views; the raw-logit center is updated after each loss. This isolates multi-view distillation and center EMA, not image ViT pretraining or teacher-parameter EMA.'},
        'metrics':{'updates':80,'trainable_parameters':54,'initial_loss':losses[0],'final_loss':losses[-1],'training_accuracy':accuracy,
            'scalar_error':scalar_error,'center_error':center_error,'probability_error':probability_error,
            'teacher_stopped':teacher.grad is None,'center_detached':not center.requires_grad},
        'charts':[{'kind':'line','title':'Submitted multi-view distillation loss','x':steps,'x_label':'SGD update','series':[{'name':'Loss','values':losses}]},
            {'kind':'line','title':'Running raw-logit center','x':steps,'x_label':'SGD update','series':[{'name':'Center L2 norm','values':center_norm}]}]}
