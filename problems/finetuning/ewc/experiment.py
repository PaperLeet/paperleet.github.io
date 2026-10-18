"""Two synthetic linear tasks demonstrate a retention/adaptation tradeoff."""
import torch


def run(penalty_fn, steps_a=60, steps_b=160):
    # Unit-variance Gaussian linear likelihood: the supplied A Fisher diagonal
    # is E[x_i^2] = [2.5, 0]. There is no sampled Fisher estimator in this lab.
    x_a = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]], dtype=torch.float64)
    y_a = 2 * x_a[:, 0]
    x_b = torch.tensor([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]], dtype=torch.float64)
    y_b = -2 * x_b[:, 0] + 2 * x_b[:, 1]
    theta_a = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([theta_a], lr=.2)
    task_a_curve = []
    for step in range(steps_a + 1):
        mse = (x_a @ theta_a - y_a).square().mean()
        if step % 10 == 0 or step == steps_a:
            task_a_curve.append([step, float(mse.detach())])
        if step < steps_a:
            optimizer.zero_grad()
            (.5 * mse).backward()
            optimizer.step()
    anchor = theta_a.detach().clone()
    fisher = torch.tensor([2.5, 0.], dtype=torch.float64)
    runs = {}
    for name, strength in (("unregularized", 0.), ("ewc", 1.)):
        theta = anchor.clone().requires_grad_()
        optimizer = torch.optim.SGD([theta], lr=.1)
        curve = []
        for step in range(steps_b + 1):
            new_mse = (x_b @ theta - y_b).square().mean()
            # This is the submitted implementation, including in every update.
            penalty = penalty_fn(theta, anchor, fisher, strength)
            objective = .5 * new_mse + penalty
            if step % 10 == 0 or step == steps_b:
                with torch.no_grad():
                    old_mse = (x_a @ theta - y_a).square().mean()
                curve.append({"step": step, "old_mse": float(old_mse),
                              "new_mse": float(new_mse.detach()),
                              "objective": float(objective.detach()), "penalty": float(penalty.detach())})
            if step < steps_b:
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
        runs[name] = {"strength": strength, "theta": theta.detach().tolist(), "curve": curve}
    final_plain = runs["unregularized"]["curve"][-1]
    final_ewc = runs["ewc"]["curve"][-1]
    return {
        "kind": "toy_experiment", "problem": "finetuning/ewc",
        "title": "Learn task B while measuring retention of task A",
        "provenance": {
            "kind": "teaching_adaptation", "source": "https://arxiv.org/abs/1612.00796v2",
            "anchor": "Section 2, Eq. (3)",
            "adaptation": "Two weights, four deterministic examples per task, supplied Gaussian-model Fisher [2.5,0]. 60 task-A SGD updates then 160 task-B updates per arm; no MNIST, Atari, LLM, replay or estimated Fisher. MSE is measured on these synthetic tasks, not held-out generalization."
        },
        "metrics": {"task_a_before_b": task_a_curve[-1][1],
                    "unregularized_old_mse": final_plain["old_mse"],
                    "unregularized_new_mse": final_plain["new_mse"],
                    "ewc_old_mse": final_ewc["old_mse"], "ewc_new_mse": final_ewc["new_mse"],
                    "initial_new_mse": runs["ewc"]["curve"][0]["new_mse"],
                    "old_mse_increase": final_ewc["old_mse"] - task_a_curve[-1][1]},
        "anchor": anchor.tolist(), "fisher_diag": fisher.tolist(), "runs": runs,
        "loss_curve": [[row["step"], row["objective"]] for row in runs["ewc"]["curve"]],
        "charts": [
            {"kind": "line", "title": "First train task A", "x_label": "A updates",
             "x": [row[0] for row in task_a_curve],
             "series": [{"name": "Task A MSE", "values": [row[1] for row in task_a_curve]}]},
            {"kind": "line", "title": "Old-task retention and new-task learning", "x_label": "B updates",
             "x": [row["step"] for row in runs["ewc"]["curve"]],
             "series": [{"name": name + " / " + metric,
                         "values": [row[metric] for row in runs[name]["curve"]]}
                        for name in ("unregularized", "ewc") for metric in ("old_mse", "new_mse")]}
        ]
    }
