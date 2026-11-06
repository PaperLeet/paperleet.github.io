"""Three rounds of tiny alternating clustering and encoder/head optimization."""
import torch
from .toy import fixture


def run(normalize_features, assign_clusters, pseudo_label_loss):
    inputs = fixture()
    encoder = torch.nn.Parameter(torch.eye(2, dtype=torch.float64))
    head = torch.nn.Parameter(torch.zeros((2, 3), dtype=torch.float64))
    optimizer = torch.optim.SGD([encoder, head], lr=.25)
    centroids = inputs[[0, 3, 6]]
    steps, losses, accuracies = [], [], []
    labels = None
    for step in range(91):
        embeddings = normalize_features(inputs @ encoder)
        if step % 30 == 0:
            with torch.no_grad():
                labels = assign_clusters(embeddings, centroids)
                centroids = torch.stack([embeddings[labels == k].mean(0) if bool((labels == k).any())
                                         else centroids[k] for k in range(3)])
        logits = embeddings @ head
        loss = pseudo_label_loss(logits, labels)
        if step % 10 == 0:
            steps.append(step); losses.append(float(loss.detach()))
            accuracies.append(float((logits.argmax(1) == labels).double().mean()))
        if step < 90:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {'provenance': 'Original nine-point, three-cluster teaching adaptation: 90 SGD updates (lr .25), a 2x2 encoder and 2x3 head, cluster refresh every 30 updates; no PCA fit, image model or semantic accuracy benchmark.',
            'charts': [{'kind': 'line', 'title': 'Alternating pseudo-label training', 'x': steps,
                        'x_label': 'SGD update', 'y_label': 'Training metric',
                        'series': [{'name': 'Pseudo-label CE', 'values': losses},
                                   {'name': 'Pseudo-label accuracy', 'values': accuracies}]}],
            'metrics': {'updates': 90, 'trainable_parameters': 10, 'cluster_refreshes': 4,
                        'initial_loss': losses[0], 'final_loss': losses[-1],
                        'initial_accuracy': accuracies[0], 'final_accuracy': accuracies[-1],
                        'occupied_clusters': len(set(labels.tolist())),
                        'encoder_motion': float((encoder.detach() - torch.eye(2,dtype=torch.float64)).norm())}}
