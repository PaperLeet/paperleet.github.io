"""Tensor comparisons with actionable location information."""
import torch
from .comparisons import diagnostic_scalar, normalize_comparison


def assert_close_located(got, exp, rtol=1e-5, atol=1e-8):
    got, exp = torch.as_tensor(got), torch.as_tensor(exp)
    if got.shape != exp.shape:
        failure = AssertionError(f"at shape: expected {tuple(exp.shape)}, got {tuple(got.shape)}")
        failure.paperleet_comparison = normalize_comparison({"kind": "shape", "expected_shape": list(exp.shape), "actual_shape": list(got.shape)})
        raise failure
    expected = exp.to(dtype=got.dtype, device=got.device)
    close = torch.isclose(got, expected, rtol=rtol, atol=atol)
    if bool(close.all()):
        return
    error = ((got.to(torch.float64) - expected.to(torch.float64)).abs() if got.dtype == torch.bool or str(got.dtype).endswith('uint8')
             else (got - expected).abs().to(torch.float64))
    error = torch.where(close, torch.full_like(error, -1), error)
    error = torch.nan_to_num(error, nan=float("inf"), posinf=float("inf"))
    flat = int(error.reshape(-1).argmax())
    index = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), got.shape)) if got.ndim else ()
    failure = AssertionError(f"max abs diff {error[index].item():.7g} at {index}: expected {expected[index].item()}, got {got[index].item()} (rtol={rtol}, atol={atol})")
    failure.paperleet_comparison = normalize_comparison({"kind": "value", "expected_shape": list(expected.shape), "actual_shape": list(got.shape),
        "index": list(index), "expected": diagnostic_scalar(expected[index].item()), "actual": diagnostic_scalar(got[index].item()),
        "absolute_error": diagnostic_scalar(error[index].item()), "rtol": rtol, "atol": atol})
    raise failure
