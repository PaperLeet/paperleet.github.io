from problems.representation.deepcluster.solution import normalize_features, assign_clusters, pseudo_label_loss
from problems.representation.deepcluster.experiment import run


def generate():
    result = run(normalize_features, assign_clusters, pseudo_label_loss)
    return {'kind': 'toy_experiment', 'title': 'DeepCluster: Learning from Cluster Pseudo-Labels',
            'problem': 'representation/deepcluster',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/1807.05520',
                           'anchor': 'Section 3: alternating clustering and representation learning', 'adaptation': result['provenance']},
            'charts': result['charts'], 'metrics': result['metrics']}
