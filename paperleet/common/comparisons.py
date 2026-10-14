"""Bounded, JSON-safe optional numerical failure diagnostics."""
import math
import re


def _shape(value):
    return (isinstance(value, list) and len(value) <= 32 and
            all(type(size) is int and 0 <= size <= 2**53 - 1 for size in value))


def _scalar(value):
    if type(value) is bool: return True
    if type(value) is int: return abs(value) <= 2**53 - 1
    if type(value) is float: return math.isfinite(value)
    return isinstance(value, str) and len(value) <= 32 and (
        value in ("NaN", "Infinity", "-Infinity") or re.fullmatch(r"-?\d+", value) is not None)


def diagnostic_scalar(value):
    if type(value) is float and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if type(value) is int and abs(value) > 2**53 - 1: return str(value)
    return value


def normalize_comparison(value):
    if not isinstance(value, dict) or value.get("kind") not in ("shape", "value"): return None
    expected, actual = value.get("expected_shape"), value.get("actual_shape")
    if not _shape(expected) or not _shape(actual): return None
    result = {"kind": value["kind"], "expected_shape": list(expected), "actual_shape": list(actual)}
    if value["kind"] == "shape":
        return result if expected != actual else None
    index = value.get("index")
    if expected != actual or not isinstance(index, list) or len(index) != len(actual): return None
    if not all(type(position) is int and 0 <= position < size for position,size in zip(index, actual)): return None
    if not all(_scalar(value.get(key)) for key in ("expected", "actual", "absolute_error")): return None
    error = value["absolute_error"]
    if (type(error) not in (int, float) or error < 0) and error != "Infinity": return None
    for key in ("rtol", "atol"):
        number = value.get(key)
        if type(number) not in (int, float) or number < 0: return None
        if type(number) is int and number > 2**53 - 1: return None
        if type(number) is float and not math.isfinite(number): return None
    result.update(index=list(index), expected=value["expected"], actual=value["actual"],
                  absolute_error=error, rtol=value["rtol"], atol=value["atol"])
    return result


def comparison_from_exception(error):
    try: return normalize_comparison(getattr(error, "paperleet_comparison", None))
    except Exception: return None  # Optional diagnostics cannot interrupt judging.
