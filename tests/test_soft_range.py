import numpy as np
import pandas as pd
import pytest
from main import yearly_to_monthly
from smoothing import FIRST_MONTH_FLOOR, FIRST_MONTH_FRACTION


def convert(values):
    source = pd.DataFrame({"year":range(1,len(values)+1), "v":values})
    result = yearly_to_monthly(source,"average").v.to_numpy()
    np.testing.assert_allclose(result.reshape(-1,12).mean(axis=1),values,atol=1e-6,rtol=0)
    epsilon=max(FIRST_MONTH_FLOOR,FIRST_MONTH_FRACTION*max(max(values),0))
    assert result[0]>0 and result[0]>=epsilon*(1-1e-8)
    return result


@pytest.mark.parametrize("values", [
 [1,4,10,20,33,48,64,79,92,98,100,100,100],
 [4,14,30,50,68,82,91,96,99,98,100,100,100],
 [8,25,48,68,83,92,97,99,99.8,98,100,100,100],
])
def test_thirteen_year_diagnostic(values):
    result=convert(values)
    assert max(values)-2 <= result.max() <= max(values)+2
    assert result.min() > 0  # No input-minimum restriction remains.
    assert np.unique(result[48:60]).size>=3


def test_gentle_data_uses_soft_range_penalty():
    result=convert([6,8,12])
    assert result.min()>5.9 and result.max()<12.3
    assert result.min()<6 or result.max()>12  # Not a hard box constraint.


@pytest.mark.parametrize("values", [
 [1,4,10,20,33,48,64,79,92,100],
 [4,14,30,50,68,82,91,96,99,100],
 [8,25,48,68,83,92,97,99,99.8,100],
])
def test_original_ten_year_range_regression(values):
    result=convert(values)
    assert max(0,result.max()-max(values)) < 0.33
    assert max(0,min(values)-result.min()) < 0.74
    # Maximum-target years remain accepted with the correct mean, not clamped.
    assert result[-12:].mean()==pytest.approx(100,abs=1e-6)
    assert result.max()>100


def test_jump_excursions_near_transition():
    result=convert([10,10,90,90])
    assert 12 <= np.argmin(result) <24
    assert 24 <= np.argmax(result) <36
    assert abs(result[24]-result[23])<20
    assert result.min()<10 and result.max()>90


def test_absolute_values_and_independent_column_scales():
    values=[1000,4000,9000]
    result=convert(values)
    assert result[0]>0
    assert result.max()>max(values)
    frame=pd.DataFrame({"year":[1,2,3],"small":[6,8,12],"large":values})
    combined=yearly_to_monthly(frame,"average")
    np.testing.assert_allclose(combined.large,result,atol=1e-5)
    np.testing.assert_allclose(combined.small,convert([6,8,12]),atol=1e-6)


@pytest.mark.parametrize("values", [[0],[-1],[-3,-2,-1],[0,0],[1e-9,2e-9],[0,100,0,100]])
def test_positive_start_even_for_nonpositive_or_tiny_targets(values):
    convert(values)
