"""Teaching adaptation: learn four paired feature vectors on CPU."""
import torch


def run(pairwise_logits, siglip_loss):
    # Four paired identities, initialized in deliberately different directions.
    images = torch.eye(4, dtype=torch.float64).requires_grad_()
    texts = torch.tensor([[.3, .9, -.2, .1], [-.4, .2, .8, .3],
                          [.8, -.3, .2, .4], [.2, .1, -.3, .7]], dtype=torch.float64).requires_grad_()
    log_scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(-1., dtype=torch.float64, requires_grad=True)
    losses, correct = [], []
    for step in range(121):
        logits = pairwise_logits(images, texts, log_scale, bias)
        loss = siglip_loss(logits)
        losses.append(loss.item())
        correct.append((logits.argmax(-1) == torch.arange(4)).to(torch.float64).mean().item())
        if step < 120:
            values = (images, texts, log_scale, bias)
            grads = torch.autograd.grad(loss, values)
            images, texts, log_scale, bias = [(x - .15 * g).detach().requires_grad_() for x, g in zip(values, grads)]
    logits = pairwise_logits(images, texts, log_scale, bias).detach()
    diagonal = logits.diag()
    negative = logits[~torch.eye(4, dtype=torch.bool)]
    return {
        "kind": "toy_experiment", "title": "SigLIP: learning four image-text pairs",
        "problem": "multimodal/siglip",
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2303.15343",
                       "anchor": "Section 3.2 and Algorithm 1",
                       "adaptation": "Optimize four synthetic paired embeddings, log scale and bias for 120 CPU steps; no image/text encoder or pretraining dataset."},
        "metrics": {"initial_loss": losses[0], "final_loss": losses[-1],
                    "initial_retrieval_accuracy": correct[0], "final_retrieval_accuracy": correct[-1],
                    "minimum_positive_logit": diagonal.min().item(),
                    "maximum_negative_logit": negative.max().item()},
        "charts": [{"kind": "line", "title": "Pairwise logistic training", "x_label": "Gradient step",
                    "x": list(range(len(losses))), "series": [{"name": "SigLIP loss", "values": losses}]},
                   {"kind": "line", "title": "Image-to-text retrieval", "x_label": "Gradient step",
                    "x": list(range(len(correct))), "series": [{"name": "Accuracy", "values": correct}]}],
    }
