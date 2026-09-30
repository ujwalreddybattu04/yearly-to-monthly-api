"""PCHIP Average must preserve means or reject unsafe results, never clip."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.interpolate import PchipInterpolator
from main import app, yearly_to_monthly

client = TestClient(app)
ADOPTION = pd.DataFrame({
    "year": range(1, 11),
    "Slow": [1,4,10,20,33,48,64,79,92,100],
    "Moderate": [4,14,30,50,68,82,91,96,99,100],
    "Fast": [8,25,48,68,83,92,97,99,99.8,100],
})

@pytest.mark.parametrize("column", ["Slow", "Moderate", "Fast"])
@pytest.mark.parametrize("fmt", ["csv", "json", "pdf"])
def test_adoption_rejected_instead_of_returning_impossible_percentages(column, fmt):
    source = ADOPTION[["year", column]]
    with pytest.raises(ValueError, match=column + "' cannot be smoothed"):
        yearly_to_monthly(source, "average")
    response = client.post("/convert", params={"mode":"average", "format":fmt},
        files={"file":("curves.csv", source.to_csv(index=False), "text/csv")})
    assert response.status_code == 400
    assert "0-100%" in response.json()["detail"]
    assert "Exit mode" in response.json()["detail"]

@pytest.mark.parametrize("values", [[0,100,0,100], [6,12,3,9], [0,1], [99,100], [-1], [101]])
def test_unsafe_results_are_rejected(values):
    source = pd.DataFrame({"year":range(len(values)), "value":values})
    with pytest.raises(ValueError, match="0-100%"):
        yearly_to_monthly(source, "average")
    response = client.post("/convert?mode=average", files={"file":("a.csv",source.to_csv(index=False))})
    assert response.status_code == 400

@pytest.mark.parametrize("values", [[6,8,12], [12,8,6], [20,40,60,80], [0,0], [100,100], [0], [100]])
def test_safe_results_preserve_means_and_pchip_calculation(values):
    source = pd.DataFrame({"year":range(len(values)), "value":values})
    result = yearly_to_monthly(source, "average")
    assert result.value.between(0,100).all()
    np.testing.assert_allclose(result.groupby("year").value.mean(), values, atol=1e-9, rtol=0)
    if len(values)>1:
        candidate = PchipInterpolator(np.arange(len(values))*12+6.5, values)(np.arange(1,len(values)*12+1)).reshape(-1,12)
        expected = candidate + (np.array(values)-candidate.mean(axis=1))[:,None]
        np.testing.assert_allclose(result.value, expected.ravel(), atol=1e-12, rtol=0)


def test_adoption_exit_stays_bounded_and_matches_endpoints():
    result = yearly_to_monthly(ADOPTION, "exit")
    assert result.iloc[:,2:].ge(0).all().all()
    assert result.iloc[:,2:].le(100).all().all()
    np.testing.assert_array_equal(result[result.month==12].iloc[:,2:], ADOPTION.iloc[:,1:])


def test_bad_second_column_is_identified_even_when_named_month():
    source = pd.DataFrame({"year":[1,2], "safe":[6,8], "month":[99,100]})
    with pytest.raises(ValueError, match="Column 'month' cannot be smoothed"):
        yearly_to_monthly(source, "average")
