from decimal import Decimal, localcontext
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import app
from smoothing import smooth_average, _verify_monthly_means, _check_magnitude_ratio


def assert_decimal_means(result, targets):
    with localcontext() as ctx:
        ctx.prec = 800
        for i, target in enumerate(targets):
            actual = sum((Decimal(str(v)) for v in result[i*12:(i+1)*12,0]), Decimal(0))/12
            expected = Decimal(str(target))
            assert abs(actual-expected)/max(abs(expected),Decimal("1e-6")) < Decimal("1e-6")


def test_exact_reproduction_rejected_before_solver(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Unsafe dynamic range must not reach solver construction")
    monkeypatch.setattr("smoothing.cp.Problem",fail)
    with pytest.raises(ValueError,match="Column 'sales'.*too wide a range"):
        smooth_average(np.array([[1.],[1e308]]),["sales"])


@pytest.mark.parametrize("fmt",["csv","json","pdf"])
def test_api_bad_column_names_whole_request(fmt):
    r=TestClient(app).post("/convert",params={"mode":"average","format":fmt},
      files={"file":("a.csv","year,normal,problem\n1,6,1\n2,8,1e308")})
    assert r.status_code==400
    assert "Column 'problem'" in r.json()["detail"]
    assert "too wide a range" in r.json()["detail"]


@pytest.mark.parametrize("large",[100000.,1e12*(1-1e-6),1e12])
def test_safe_and_threshold_ranges(large):
    targets=np.array([[1.],[large]])
    result=smooth_average(targets,["sales"])
    assert_decimal_means(result,targets[:,0])


def test_above_threshold_and_ratio_overflow():
    for values in [[1.,1e12*(1+1e-6)],[1e-308,1e308]]:
        with pytest.raises(ValueError,match="Column 'wide'.*too wide a range"):
            smooth_average(np.array(values).reshape(-1,1),["wide"])


@pytest.mark.parametrize("values",[[0.,0.],[0.,6.],[0.],[1e308]])
def test_zero_one_nonzero_and_constant_fast_paths(values):
    _check_magnitude_ratio(np.array(values),"v")
    result=smooth_average(np.array(values).reshape(-1,1),["v"])
    assert_decimal_means(result,values)


def test_verifier_detects_corruption_and_cancellation():
    with pytest.raises(ValueError,match="year index 0"):
        _verify_monthly_means(np.full(12,2.),np.array([1.]),"v")
    # Ordinary float addition can discard the small residual completely.
    corrupted=np.array([1e308,1.,-1e308]+[0.]*9)
    with pytest.raises(ValueError,match="year index 0"):
        _verify_monthly_means(corrupted,np.array([0.]),"v")


@pytest.mark.parametrize("targets",[np.array([[6.]]),np.array([[6.],[6.]]),np.array([[6.],[8.]])])
def test_verifier_runs_on_every_return_path(monkeypatch,targets):
    calls=[]
    import smoothing
    original=smoothing._verify_monthly_means
    def record(values,expected,name):
        calls.append(name)
        original(values,expected,name)
    monkeypatch.setattr(smoothing,"_verify_monthly_means",record)
    smooth_average(targets,["v"])
    assert calls==["v"]


def test_verifier_does_not_overflow_on_max_scale_constant():
    _verify_monthly_means(np.full(12,1e308),np.array([1e308]),"v")
