"""Actionable comparison diagnostics must preserve actual assertion outcomes."""
import json
import math

import pytest
import torch

from paperleet.common.asserts import assert_close_located as close
from paperleet.common.comparisons import comparison_from_exception, normalize_comparison


def failure(got, expected, **kw):
    with pytest.raises(AssertionError) as caught: close(got, expected, **kw)
    result = comparison_from_exception(caught.value)
    assert result is not None
    json.dumps(result, allow_nan=False)
    return result


def test_value_mismatch_coordinates_and_tolerances_are_explicit():
    result = failure(torch.tensor([[1., 2.], [9., 4.]]), torch.tensor([[1., 2.], [3., 4.]]), rtol=1e-4, atol=1e-6)
    assert result == {'kind':'value','expected_shape':[2,2],'actual_shape':[2,2], 'index':[1,0],
                      'expected':3.,'actual':9.,'absolute_error':6.,'rtol':1e-4,'atol':1e-6}
    scalar = failure(torch.tensor(.25), torch.tensor(.5))
    assert scalar['index'] == [] and scalar['expected_shape'] == []


def test_shape_mismatch_is_reported_without_comparing_or_broadcasting():
    assert failure(torch.zeros(2,3), torch.zeros(3,2)) == {'kind':'shape','expected_shape':[3,2],'actual_shape':[2,3]}
    assert failure(torch.zeros(0,3), torch.zeros(0,4))['kind'] == 'shape'
    close(torch.zeros(0,3), torch.zeros(0,3))


@pytest.mark.parametrize('actual,expected,token', [(float('nan'),1.,'NaN'),(float('inf'),1.,'Infinity'),(-float('inf'),1.,'-Infinity'),(float('nan'),float('nan'),'NaN')])
def test_nonfinite_mismatches_remain_failures_and_strict_json(actual, expected, token):
    result = failure(torch.tensor(actual), torch.tensor(expected))
    assert result['actual'] == token and result['absolute_error'] == 'Infinity'
    close(torch.tensor(float('inf')), torch.tensor(float('inf')))


def test_boolean_and_uint8_differences_do_not_wrap_or_raise_subtraction_errors():
    assert failure(torch.tensor([False]), torch.tensor([True]))['absolute_error'] == 1
    assert failure(torch.tensor([0],dtype=torch.uint8),torch.tensor([255],dtype=torch.uint8))['absolute_error'] == 255


def test_large_integer_values_keep_their_exact_digits():
    result = failure(torch.tensor([2**60],dtype=torch.int64),torch.tensor([0],dtype=torch.int64),rtol=0,atol=0)
    assert result['actual'] == str(2**60)


def test_optional_diagnostics_cannot_change_acceptance_or_crash_on_bad_attributes():
    class ErrorWithBadAttribute(Exception):
        @property
        def paperleet_comparison(self): raise RuntimeError('untrusted attribute')
    assert comparison_from_exception(ErrorWithBadAttribute()) is None
    assert comparison_from_exception(AssertionError('legacy trace')) is None
    base={'kind':'value','expected_shape':[2],'actual_shape':[2],'index':[1], 'expected':1.,'actual':2.,'absolute_error':1.,'rtol':1e-5,'atol':1e-8}
    for bad in [None, [], {}, {**base,'index':[2]}, {**base,'index':[True]}, {**base,'expected_shape':[3]}, {**base,'actual':math.inf}, {**base,'absolute_error':-1}, {**base,'rtol':math.nan}, {**base,'expected_shape':[1]*33}, {**base,'actual':'<script>alert(1)</script>'}]:
        assert normalize_comparison(bad) is None
