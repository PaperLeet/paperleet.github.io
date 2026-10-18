from torch import Tensor


def continual_metrics(scores: Tensor, baseline: Tensor) -> dict[str, Tensor]:
    """Compute accuracy, transfer and peak forgetting after sequential training.

    scores[i,j] is task j's accuracy after training stage i, shape [T,T].
    baseline[j] is task j's accuracy before any training, shape [T], T>=1.
    Both are CPU float32/64 with matching dtype. Used entries are in [0,1].
    The upper triangle beyond the first superdiagonal and baseline[0] are
    unused and may contain NaN. Preserve both inputs. No gradients required.

    Return exactly acc/bwt/fwt/forgetting, each a scalar Tensor with the
    input dtype/device. Average tasks equally, not by test-set size.
    For T=1, acc=scores[0,0] and all other metrics are zero.
    See the task for the historical-peak convention, distinct from -BWT.
    """
    raise NotImplementedError
