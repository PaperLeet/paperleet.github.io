# Bug: Ignore teacher temperature.
"""DINO centered teacher distributions, multi-view distillation and center EMA."""
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def teacher_probabilities(teacher: Tensor, center: Tensor, temperature: float = .04) -> Tensor:
    """Detached probabilities for two teacher global views [2,B,K], center [K]."""
    _teacher(teacher,center);_positive(temperature,'teacher temperature')
    return F.softmax((teacher.detach()-center.detach()),dim=-1)


def dino_loss(student: Tensor, teacher: Tensor, center: Tensor, student_temperature: float = .1, teacher_temperature: float = .04) -> Tensor:
    """Average teacher-to-student cross-entropy over all 2*(V-1) unequal-view pairs."""
    _teacher(teacher,center);_positive(student_temperature,'student temperature')
    _tensor(student)
    if student.ndim!=3 or student.shape[0]<2 or student.shape[1:]!=teacher.shape[1:] or student.dtype!=teacher.dtype or student.device!=teacher.device:
        raise ValueError('student must be matching [V,B,K], V>=2')
    probabilities=teacher_probabilities(teacher,center,teacher_temperature)
    logp=F.log_softmax(student/student_temperature,dim=-1)
    terms=[]
    for t in range(2):
        for v in range(student.shape[0]):
            if v!=t:
                terms.append(-(probabilities[t]*logp[v]).sum(dim=-1).mean())
    return torch.stack(terms).mean()


def update_center(center: Tensor, teacher: Tensor, momentum: float = .9) -> Tensor:
    """Detached center EMA from raw teacher logits, averaged across both views and examples."""
    _teacher(teacher,center)
    if type(momentum) not in (int,float) or not math.isfinite(momentum) or not 0<=momentum<=1:
        raise ValueError('momentum must be a finite Python number in [0,1]')
    batch_center=teacher.detach().mean(dim=(0,1))
    return (momentum*center.detach()+(1-momentum)*batch_center).clone()


def _tensor(value):
    if not isinstance(value,Tensor) or value.numel()==0 or value.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(value).all()):
        raise ValueError('expected nonempty finite float32/float64 tensor')


def _teacher(teacher,center):
    _tensor(teacher);_tensor(center)
    if teacher.ndim!=3 or teacher.shape[0]!=2 or center.shape!=(teacher.shape[-1],):
        raise ValueError('teacher must be [2,B,K] and center [K]')
    if teacher.dtype!=center.dtype or teacher.device!=center.device:
        raise ValueError('teacher and center must share dtype and device')


def _positive(value,name):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be a finite positive Python number')
