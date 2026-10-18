from paperleet.common.asserts import assert_close_diagnostic as close
import pytest
import torch


KEYS = {"acc", "bwt", "fwt", "forgetting"}


def scalar_oracle(scores, baseline):
    """Use Python lists and explicit stage/task loops, independent of tensor reductions."""
    rows, before = scores.tolist(), baseline.tolist()
    tasks = len(rows)
    result = {"acc": sum(rows[-1]) / tasks, "bwt": 0., "fwt": 0., "forgetting": 0.}
    if tasks > 1:
        for task in range(tasks - 1):
            result["bwt"] += (rows[-1][task] - rows[task][task]) / (tasks - 1)
            peak = max(rows[stage][task] for stage in range(task, tasks))
            result["forgetting"] += (peak - rows[-1][task]) / (tasks - 1)
        for task in range(1, tasks):
            result["fwt"] += (rows[task - 1][task] - before[task]) / (tasks - 1)
    return result


def check_values(got, expected, like):
    assert isinstance(got, dict) and set(got) == KEYS
    for name, value in expected.items():
        close(got[name], like.new_tensor(value), rtol=1e-6, atol=1e-7)


@pytest.mark.t0
def test_scalar_dictionary_contract(impl):
    """Each metric is a scalar Tensor preserving the input dtype and device."""
    for dtype in (torch.float32, torch.float64):
        scores = torch.tensor([[.8, .2], [.6, .9]], dtype=dtype)
        got = impl.continual_metrics(scores, torch.tensor([.1, .1], dtype=dtype))
        assert isinstance(got, dict) and set(got) == KEYS
        for value in got.values():
            assert isinstance(value, torch.Tensor)
            assert value.shape == () and value.dtype == dtype and value.device == scores.device
        check_values(got, {"acc": .75, "bwt": -.2, "fwt": .1, "forgetting": .2}, scores)


@pytest.mark.t1
def test_hand_calculated_positive_transfer_and_peak_drop(impl):
    """A task can improve over its diagonal while still losing accuracy from a later peak."""
    scores = torch.tensor([[.5, .2, float('nan')], [.9, .6, .3], [.7, .8, .9]], dtype=torch.float64)
    baseline = torch.tensor([float('nan'), .1, .15], dtype=scores.dtype)
    # ACC=(.7+.8+.9)/3; BWT=(.2+.2)/2; FWT=(.1+.15)/2;
    # forgetting=((.9-.7)+(.8-.8))/2, despite positive BWT.
    check_values(impl.continual_metrics(scores, baseline),
                 {"acc": .8, "bwt": .2, "fwt": .125, "forgetting": .1}, scores)


@pytest.mark.t1
def test_independent_multistage_oracle(impl):
    """Separate scalar stage/task loops cover unequal diagonals, gains, losses and baselines."""
    rows = [[.8, .3, .99, .99], [.6, .7, .2, .99],
            [.9, .5, .6, .4], [.7, .8, .4, .9]]
    for dtype in (torch.float32, torch.float64):
        scores = torch.tensor(rows, dtype=dtype)
        baseline = torch.tensor([.2, .1, .35, .25], dtype=dtype)
        check_values(impl.continual_metrics(scores, baseline), scalar_oracle(scores, baseline), scores)


@pytest.mark.t2
def test_single_task_endpoint(impl):
    """With no old or future task, transfer and forgetting are zero rather than undefined."""
    for dtype in (torch.float32, torch.float64):
        scores = torch.tensor([[.625]], dtype=dtype)
        check_values(impl.continual_metrics(scores, torch.tensor([float('nan')], dtype=dtype)),
                     {"acc": .625, "bwt": 0., "fwt": 0., "forgetting": 0.}, scores)


@pytest.mark.t2
def test_unused_future_cells_and_baseline_are_ignored(impl):
    """Only immediately-pretraining cells may inform FWT; unrelated future evaluations cannot leak."""
    scores = torch.tensor([[.5, .9, .98, .97], [.6, .4, .8, .96],
                           [.7, .5, .3, .7], [.8, .6, .4, .9]], dtype=torch.float64)
    baseline = torch.tensor([.99, .2, .3, .4], dtype=scores.dtype)
    expected = scalar_oracle(scores, baseline)
    changed = scores.clone()
    for i in range(4):
        for j in range(i + 2, 4):
            changed[i, j] = float('nan')
    baseline[0] = float('nan')
    check_values(impl.continual_metrics(changed, baseline), expected, scores)
    # High pretraining performance is not part of the post-learning peak.
    close(impl.continual_metrics(changed, baseline)["forgetting"], scores.new_tensor(0.))


@pytest.mark.t2
def test_taskwise_offsets_and_baseline_effect(impl):
    """A shared taskwise offset cancels in every difference; baseline affects only FWT."""
    scores = torch.tensor([[.4, .2, .1], [.5, .45, .3], [.35, .55, .6]], dtype=torch.float64)
    baseline = torch.tensor([.1, .15, .2], dtype=scores.dtype)
    offsets = torch.tensor([.05, .1, .15], dtype=scores.dtype)
    original = impl.continual_metrics(scores, baseline)
    shifted = impl.continual_metrics(scores + offsets, baseline + offsets)
    close(shifted["acc"], original["acc"] + offsets.mean())
    for key in ("bwt", "fwt", "forgetting"):
        close(shifted[key], original[key])
    changed = impl.continual_metrics(scores, baseline + .1)
    close(changed["fwt"], original["fwt"] - .1)
    for key in ("acc", "bwt", "forgetting"):
        close(changed[key], original[key])


@pytest.mark.t2
def test_strided_inputs_preservation_and_repeatability(impl):
    """Evaluation is stateless and accepts noncontiguous views without modifying either input."""
    scores = (torch.arange(36, dtype=torch.float64).reshape(6, 6) / 40)[::2, ::2]
    baseline = torch.tensor([.1, .99, .2, .99, .3, .99], dtype=scores.dtype)[::2]
    before_scores, before_baseline = scores.clone(), baseline.clone()
    first = impl.continual_metrics(scores, baseline)
    check_values(first, scalar_oracle(scores, baseline), scores)
    second = impl.continual_metrics(scores, baseline)
    for key in KEYS:
        close(first[key], second[key])
    close(scores, before_scores)
    close(baseline, before_baseline)


@pytest.mark.t3
def test_sequential_training_evaluation(impl, artifact):
    """Learner metrics evaluate held-out predictions after each real sequential training stage."""
    from problems.foundations.continual_metrics.experiment import run
    artifact('sequential_evaluation', run(impl.continual_metrics))
