from problems.representation.vae.solution import reparameterize, gaussian_kl, negative_elbo
from problems.representation.vae.experiment import run


def generate():
    result = run(reparameterize, gaussian_kl, negative_elbo)
    return {'kind': 'toy_experiment', 'title': 'VAE: Reparameterization and the Bernoulli ELBO',
            'problem': 'representation/vae',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/1312.6114v11#S3',
                           'anchor': 'Section 2.2, Equation (3): variational lower bound; Section 3, Equations (9)-(10): diagonal Gaussian, pathwise sampling and analytic KL; Appendices B and C.1: Gaussian KL derivation and Bernoulli decoder',
                           'adaptation': 'A linear two-dimensional Gaussian encoder and Bernoulli decoder train for 120 Adam updates on two complementary six-bit patterns. Curves report sampled negative ELBO, NLL and KL; these are not exact marginal likelihoods, MNIST results, or proof of a useful latent representation.'},
            'charts': result['charts'], 'metrics': result['metrics']}
