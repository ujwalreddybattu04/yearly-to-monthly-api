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


def test_gentle_data_matches_unpenalized_optimum():
    result=convert([6,8,12])
    # Independent equality-constrained least-curvature solution (no penalty).
    n=36
    d=np.diff(np.eye(n), n=2, axis=0)
    a=np.kron(np.eye(3),np.ones((1,12))/12)
    kkt=np.block([[2*d.T@d,a.T],[a,np.zeros((3,3))]])
    reference=np.linalg.solve(kkt,np.r_[np.zeros(n),[6,8,12]])[:n]
    assert reference[0]>FIRST_MONTH_FRACTION*12
    np.testing.assert_allclose(result,reference,atol=1e-5)


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
