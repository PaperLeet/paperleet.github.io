"""Optimize one student patch table against a fixed centered teacher distribution."""
import math
import torch


def diagnostic(student, teacher, mask, center, student_temp=.7, teacher_temp=.5):
    s, t, m, c = student.detach().tolist(), teacher.detach().tolist(), mask.tolist(), center.detach().tolist()
    def probabilities(row):
        peak = max(row); p = [math.exp(v-peak) for v in row]; return [v/sum(p) for v in p]
    ce, entropy = 0., 0.
    for b in range(len(s)):
        count = max(sum(m[b]), 1)
        for n in range(len(s[b])):
            if m[b][n]:
                q = probabilities([(v-c[k])/teacher_temp for k, v in enumerate(t[b][n])])
                scaled = [v/student_temp for v in s[b][n]]
                peak = max(scaled); logz = peak + math.log(sum(math.exp(v-peak) for v in scaled))
                ce -= sum(q[k]*(scaled[k]-logz) for k in range(len(q))) / count / len(s)
                entropy -= sum(v*math.log(v) for v in q) / count / len(s)
    return ce, entropy, ce-entropy


def run(teacher_patch_targets, ibot_patch_loss):
    index = torch.arange(48, dtype=torch.float64).reshape(3, 4, 4)
    teacher = ((index*.71).sin()*.6).requires_grad_()
    center = torch.tensor([.15, -.2, .05, .1], dtype=teacher.dtype, requires_grad=True)
    student = ((index*.23).cos()*.3).requires_grad_()
    original_teacher, original_center, original_student = teacher.detach().clone(), center.detach().clone(), student.detach().clone()
    mask = torch.tensor([[True, False, False, False], [True, True, False, True], [False, False, False, False]])
    optimizer = torch.optim.Adam([student, teacher, center], lr=.08)
    steps, losses, divergences = [], [], []
    for step in range(121):
        loss = ibot_patch_loss(student, teacher, mask, center, .7, .5)
        if step % 10 == 0:
            steps.append(step); losses.append(float(loss.detach()))
            divergences.append(diagnostic(student, original_teacher, mask, original_center)[2])
        if step < 120:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    ce, entropy, divergence = diagnostic(student, original_teacher, mask, original_center)
    targets = teacher_patch_targets(teacher, center, .5)
    return {'kind': 'toy_experiment', 'title': 'iBOT: match a centered soft patch teacher',
        'problem': 'representation/ibot',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2111.07832',
            'anchor': 'Masked patch self-distillation and official main_ibot.py iBOTLoss.forward',
            'adaptation': 'A 3-by-4-by-4 student logit table takes 120 Adam updates, lr .08, against fixed teacher logits and center. Student/teacher temperatures .7/.5 expose soft targets. No backbone, class-token objective, multicrop, EMA update, or benchmark claim.'},
        'metrics': {'initial_kl': divergences[0], 'final_kl': divergence, 'final_loss': losses[-1],
                    'oracle_ce': ce, 'teacher_entropy': entropy, 'updates': 120,
                    'teacher_drift': float((teacher.detach()-original_teacher).abs().max()),
                    'center_drift': float((center.detach()-original_center).abs().max()),
                    'visible_student_drift': float((student.detach()-original_student)[~mask].abs().max())},
        'charts': [
            {'kind': 'line', 'title': 'Actual patch self-distillation', 'x_label': 'Step', 'x': steps,
             'series': [{'name': 'Submitted CE', 'values': losses},
                        {'name': 'Fixed teacher entropy floor', 'values': [entropy for _ in steps]}]},
            {'kind': 'line', 'title': 'Independent image-balanced KL to the original teacher', 'x_label': 'Step', 'x': steps,
             'series': [{'name': 'KL divergence', 'values': divergences}]},
            {'kind': 'bar', 'title': 'First masked patch probabilities', 'x_label': 'Prototype', 'x': ['1', '2', '3', '4'],
             'series': [{'name': 'Teacher', 'values': targets[0, 0].tolist()},
                        {'name': 'Student', 'values': (student[0, 0]/.7).softmax(-1).detach().tolist()}]},
        ]}
