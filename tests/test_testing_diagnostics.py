"""Numerical feedback must enrich errors without changing Torch acceptance."""
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from paperleet._vendor.torch import as_torch
from paperleet.common.comparisons import comparison_from_exception


@pytest.fixture(params=['native', 'browser'])
def comparison_api(request):
    context = nullcontext(torch) if request.param == 'native' else as_torch()
    with context as api:
        namespace = {'__package__': 'paperleet.common'}
        source = Path(__file__).resolve().parents[1] / 'paperleet/common/asserts.py'
        exec(compile(source.read_text(), str(source), 'exec'), namespace)
        yield api, namespace['assert_close_diagnostic']


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_default_tolerances_report_a_failing_coordinate(comparison_api, dtype_name):
    api, close = comparison_api
    dtype = getattr(api, dtype_name)
    expected = api.tensor([1e8, 0.], dtype=dtype)
    actual = api.tensor([1e8 + 8., .01], dtype=dtype)
    with pytest.raises(AssertionError) as caught:
        close(actual, expected)
    result = comparison_from_exception(caught.value)
    assert result['index'] == [1], 'The larger absolute difference is allowed by rtol'
    assert result['expected'] == 0.
    assert result['actual'] == pytest.approx(.01)
    assert (result['rtol'], result['atol']) == ((1.3e-6, 1e-5) if dtype_name == 'float32' else (1e-7, 1e-7))


def test_custom_tolerances_and_nonfinite_values(comparison_api):
    api, close = comparison_api
    with pytest.raises(AssertionError) as caught:
        close(api.tensor([1., 2.]), api.tensor([1., 3.]), rtol=0., atol=.25)
    result = comparison_from_exception(caught.value)
    assert result['index'] == [1] and result['actual'] == 2. and result['expected'] == 3.
    assert result['rtol'] == 0. and result['atol'] == .25
    with pytest.raises(AssertionError) as caught:
        close(api.tensor(float('nan')), api.tensor(1.))
    assert comparison_from_exception(caught.value)['actual'] == 'NaN'


def test_shape_dtype_and_equal_nan_contracts_remain_intact(comparison_api):
    api, close = comparison_api
    with pytest.raises(AssertionError) as caught:
        close(api.zeros(2, 3), api.zeros(3, 2))
    assert comparison_from_exception(caught.value)['kind'] == 'shape'
    with pytest.raises(AssertionError) as caught:
        close(api.ones(2, dtype=api.float32), api.ones(2, dtype=api.float64))
    assert comparison_from_exception(caught.value) is None
    close(api.tensor(float('nan')), api.tensor(float('nan')), equal_nan=True)
    with pytest.raises(AssertionError) as caught:
        close(api.tensor([float('nan'), 2.]), api.tensor([float('nan'), 3.]), equal_nan=True)
    assert comparison_from_exception(caught.value) is None


def test_original_exception_is_reraised_and_diagnostic_errors_are_optional(comparison_api, monkeypatch):
    api, close = comparison_api
    original = AssertionError('original framework diagnostic')
    def reject(*args, **kwargs):
        raise original
    monkeypatch.setattr(api.testing, 'assert_close', reject)
    with pytest.raises(AssertionError) as caught:
        close(api.tensor([1.]), api.tensor([2.]))
    assert caught.value is original and str(caught.value) == 'original framework diagnostic'
    assert comparison_from_exception(caught.value)['actual'] == 1.
    close.__globals__['assert_close_located'] = lambda *args, **kwargs: 1/0
    with pytest.raises(AssertionError) as caught:
        close(api.tensor([3.]), api.tensor([4.]))
    assert caught.value is original


def test_passing_values_and_invalid_options_follow_torch(comparison_api):
    api, close = comparison_api
    close(api.tensor([1e-6]), api.tensor([0.]))
    close(api.tensor([.1]), api.tensor([0.]), rtol=0., atol=.2)
    with pytest.raises(ValueError):
        close(api.ones(1), api.zeros(1), rtol=.1)
