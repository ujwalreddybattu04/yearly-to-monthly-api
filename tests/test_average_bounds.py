"""Unrestricted quadratic smoothing preserves yearly means."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from main import app, yearly_to_monthly

client = TestClient(app)
ADOPTION = pd.DataFrame({
    "year": range(1, 11),
    "Slow": [1,4,10,20,33,48,64,79,92,100],
    "Moderate": [4,14,30,50,68,82,91,96,99,100],
    "Fast": [8,25,48,68,83,92,97,99,99.8,100],
})

def check(source):
    result = yearly_to_monthly(source, "average")
    np.testing.assert_allclose(result.groupby("year").mean().iloc[:,1:],
                               source.iloc[:,1:], atol=1e-6, rtol=0)
    return result


def test_adoption_mean_exact_and_curved():
    result = check(ADOPTION)
    assert result.iloc[:,2:].max().max() > 100
    assert result[result.year==5].iloc[:,2:].nunique().ge(3).all()
    smooth = np.square(np.diff(result.iloc[:,2:].to_numpy(), n=2, axis=0)).sum()
    flat = np.square(np.diff(np.repeat(ADOPTION.iloc[:,1:].to_numpy(),12,axis=0), n=2, axis=0)).sum()
    assert smooth < flat


@pytest.mark.parametrize("fmt", ["csv", "json", "pdf"])
def test_adoption_all_formats(fmt):
    from io import BytesIO, StringIO
    from pypdf import PdfReader
    from outputs import formatted_monthly
    expected = formatted_monthly(check(ADOPTION))
    response = client.post("/convert", params={"mode":"average", "format":fmt},
        files={"file":("curves.csv", ADOPTION.to_csv(index=False), "text/csv")})
    assert response.status_code == 200, response.text
    if fmt == "csv":
        assert pd.read_csv(StringIO(response.text)).to_dict("records") == expected.to_dict("records")
    elif fmt == "json":
        assert response.json() == expected.to_dict("records")
    else:
        assert response.content.startswith(b"%PDF")
        text = "\n".join(p.extract_text() for p in PdfReader(BytesIO(response.content)).pages)
        assert all(v in text for v in expected.iloc[:,2:].to_numpy().ravel())


@pytest.mark.parametrize("values", [[0,100,0,100], [6,12,3,9], [0,1], [99,100], [6,8,12], [0], [100], [6]])
def test_valid_targets_always_succeed(values):
    source = pd.DataFrame({"year":range(len(values)), "value":values})
    check(source)
    response = client.post("/convert?mode=average&format=json", files={"file":("a.csv",source.to_csv(index=False))})
    assert response.status_code == 200


@pytest.mark.parametrize("value", [-1,150,1000])
def test_absolute_targets_are_accepted(value):
    response = client.post("/convert?mode=average&units=number&format=json", files={"file":("a.csv",f"year,value\n1,{value}")})
    assert response.status_code == 200
    assert all(row["value"] == value for row in response.json())


def test_sparse_thousand_years_ten_columns():
    from time import perf_counter
    t = np.linspace(0, 12*np.pi, 1000)
    source = pd.DataFrame({"year":range(1000), **{f"v{i}":50+40*np.sin(t+i/10) for i in range(10)}})
    started = perf_counter()
    check(source)
    elapsed = perf_counter()-started
    print(f"1000 years x 10 columns: {elapsed:.3f}s")
    assert elapsed < 30


def test_adoption_exit_stays_bounded_and_matches_endpoints():
    result = yearly_to_monthly(ADOPTION, "exit")
    np.testing.assert_array_equal(result[result.month==12].iloc[:,2:], ADOPTION.iloc[:,1:])
