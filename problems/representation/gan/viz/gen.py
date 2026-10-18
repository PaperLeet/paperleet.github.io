from problems.representation.gan.solution import discriminator_loss, generator_loss
from problems.representation.gan.experiment import run


def generate():
    result = run(discriminator_loss, generator_loss)
    return {'kind': 'toy_experiment', 'title': 'GAN: Stable Adversarial and Non-saturating Losses', 'problem': 'representation/gan',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/1406.2661v1#S3',
                           'anchor': 'Section 3, Equation (1), Algorithm 1 and the non-saturating generator paragraph', 'adaptation': 'Real data are Gaussian with mean 2 and variance 1. The generator is `G(z)=z+shift`, and D is an affine logit. The harness runs 240 actual D/G update pairs, batch 64, Adam lr=0.03 for both, seed=179, float64. Your losses drive backward passes with G detached during D steps and D parameters frozen during G steps. Charts show the learned location and both losses. Acceptance requires final location error<0.5. This does not reproduce image quality, FID or general GAN convergence.'},
            'charts': result['charts'], 'metrics': result['metrics']}
