from io import BytesIO, StringIO
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader
from main import app, yearly_to_monthly

client = TestClient(app)

@pytest.mark.parametrize("mode", ["average", "exit"])
@pytest.mark.parametrize("fmt", ["csv", "json", "pdf"])
def test_number_units(mode, fmt):
    r = client.post("/convert", params={"mode":mode,"format":fmt,"units":"number"},
                    files={"file":("a.csv","year,sales\n1,500\n2,800\n3,1000")})
    assert r.status_code == 200
    expected = yearly_to_monthly(pd.DataFrame({"year":[1,2,3],"sales":[500,800,1000]}),mode)
    if fmt == "json":
        assert all(isinstance(row["sales"],(int,float)) for row in r.json())
        np.testing.assert_allclose([row["sales"] for row in r.json()], expected.sales)
    elif fmt == "csv":
        assert "%" not in r.text
        np.testing.assert_allclose(pd.read_csv(StringIO(r.text)).sales, expected.sales)
    else:
        text = "\n".join(p.extract_text() for p in PdfReader(BytesIO(r.content)).pages)
        assert "Monthly values" in text
        assert "%" not in text


def test_scale_invariance_and_negative_values():
    source = pd.DataFrame({"year":[1,2,3],"v":[6,8,12]})
    reference = yearly_to_monthly(source,"average").v.to_numpy()
    for scale in [1e-3,1e150,3]:
        scaled = source.copy(); scaled.v = scaled.v * scale
        actual = yearly_to_monthly(scaled,"average").v.to_numpy() / scale
        np.testing.assert_allclose(actual,reference,atol=1e-7,rtol=1e-8)


def test_no_bound_clipping_and_mean_exact():
    source = pd.DataFrame({"year":[1,2,3,4],"v":[0,100,0,100]})
    actual = yearly_to_monthly(source,"average")
    assert actual.v.min()<0 and actual.v.max()>100
    np.testing.assert_allclose(actual.groupby("year").v.mean(),source.v,atol=1e-6)


def test_invalid_units():
    assert client.post("/convert?mode=average&units=other",files={"file":("a.csv","year,v\n1,6")}).status_code==422
