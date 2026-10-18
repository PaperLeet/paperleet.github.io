"""A small measured evaluation stream, not a reproduction of GEM training."""
import math
import torch
from paperleet.common.asserts import assert_close_diagnostic as close


def _points(count, offset):
    angles = torch.arange(count, dtype=torch.float64) * (2 * math.pi / count) + offset
    return torch.stack([angles.cos(), angles.sin()], dim=1)


def _metrics_from_counts(rows, baseline):
    tasks = len(rows)
    result = {"acc": sum(rows[-1]) / tasks, "bwt": 0., "fwt": 0., "forgetting": 0.}
    if tasks > 1:
        result["bwt"] = sum(rows[-1][j] - rows[j][j] for j in range(tasks - 1)) / (tasks - 1)
        result["fwt"] = sum(rows[j - 1][j] - baseline[j] for j in range(1, tasks)) / (tasks - 1)
        result["forgetting"] = sum(max(rows[i][j] for i in range(j, tasks)) - rows[-1][j]
                                   for j in range(tasks - 1)) / (tasks - 1)
    return result


def run(solve):
    train_x, test_x = _points(64, .013), _points(80, .031)
    directions = torch.tensor([[1., .2], [.2, 1.], [-1., .2]], dtype=torch.float64)
    train_targets = (train_x @ directions.T > 0).to(torch.float64)
    test_targets = test_x @ directions.T > 0
    weights = torch.zeros(2, dtype=torch.float64)

    def evaluate():
        prediction = (test_x @ weights >= 0).unsqueeze(1)
        return (prediction == test_targets).to(torch.float64).mean(dim=0)

    baseline = evaluate()
    rows, measured = [], {key: [] for key in ("acc", "bwt", "fwt", "forgetting")}
    for task in range(3):
        for _ in range(36):
            # Actual full-batch SGD for binary logistic regression, with its
            # analytical gradient. No test examples enter these updates.
            prediction = torch.sigmoid(train_x @ weights)
            gradient = train_x.T @ (prediction - train_targets[:, task]) / train_x.shape[0]
            weights = weights - .8 * gradient
        rows.append(evaluate())
        current = torch.stack(rows)[:, :task + 1]
        got = solve(current, baseline[:task + 1])
        expected = _metrics_from_counts(current.tolist(), baseline[:task + 1].tolist())
        assert set(got) == set(expected)
        for key, value in expected.items():
            close(got[key], current.new_tensor(value), rtol=1e-7, atol=1e-9)
            measured[key].append(float(got[key]))
        assert float(rows[-1][task]) >= .9, "The currently trained task must actually be learned"
    assert measured["forgetting"][-1] > .25, "This deliberately conflicting task stream must expose forgetting"
    accuracy = torch.stack([baseline] + rows)
    return {
        "provenance": "Measured sequential training of a two-weight logistic classifier: 3 synthetic tasks, 64 train and 80 separate test inputs per task, 36 SGD steps/task, learning rate 0.8, zero initialization, deterministic circle grids (no RNG). Learner metrics are called after every stage. No GEM optimization, replay, LLM or paper-benchmark claim.",
        "charts": [
            {"kind": "line", "title": "Held-out accuracy during sequential training",
             "x": [0, 1, 2, 3], "x_label": "completed training stages", "y_label": "accuracy",
             "series": [{"name": "Task " + str(j + 1), "values": accuracy[:, j].tolist()} for j in range(3)]},
            {"kind": "line", "title": "Learner evaluation metrics",
             "x": [1, 2, 3], "x_label": "completed training stages", "y_label": "metric",
             "series": [{"name": key.upper() if key != "forgetting" else "Peak forgetting", "values": values}
                        for key, values in measured.items()]},
        ],
        "metrics": {**{key: values[-1] for key, values in measured.items()},
                    "train_examples_per_task": 64, "test_examples_per_task": 80, "sgd_updates": 108},
        "evaluation_matrix": torch.stack(rows).tolist(),
        "baseline": baseline.tolist(),
    }
