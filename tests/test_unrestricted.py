from io import BytesIO, StringIO
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader
from main import app, yearly_to_monthly

client = TestClient(app)

def test_units_removed_from_schema_and_percent_output_retained():
    schema=client.get("/openapi.json").json()
    assert "Units" not in schema["components"]["schemas"]
    for method in ["get","post"]:
        assert {p["name"] for p in schema["paths"]["/convert"][method]["parameters"]} == {"mode","format"}
    response=client.post("/convert?mode=average&format=json",files={"file":("a.csv","year,value\n1,6")})
    assert response.status_code==200
    assert response.json()[0]["value"]=="6%"


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

