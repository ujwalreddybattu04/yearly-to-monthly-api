from io import BytesIO, StringIO

import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from main import Mode, app, convert, yearly_to_monthly


client = TestClient(app)


def upload(content, mode="average", filename="yearly.csv"):
    return client.post(
        "/convert",
        params={"mode": mode},
        files={"file": (filename, content, "text/csv")},
    )


def converted(content, mode="average"):
    response = upload(content, mode)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == (
        f'attachment; filename="monthly_{mode}.csv"'
    )
    return pd.read_csv(StringIO(response.text))


def test_average_single_year():
    result = converted("year,value\n2022,1200\n")
    assert len(result) == 12
    assert result["month"].tolist() == list(range(1, 13))
    assert result["year"].eq(2022).all()
    assert result["value"].eq(100).all()
    assert result["value"].sum() == 1200


def test_average_two_years_sorted():
    result = converted("year,value\n2023,1440\n2022,1200\n")
    assert len(result) == 24
    assert result["year"].tolist() == [2022] * 12 + [2023] * 12
    assert result["month"].tolist() == list(range(1, 13)) * 2
    assert result["value"].tolist() == [100] * 12 + [120] * 12


def test_exit_example():
    result = converted("year,value\n2023,1440\n2022,1200\n", "exit")
    assert len(result) == 24
    assert result["value"].tolist() == [1200] * 12 + list(range(1220, 1441, 20))


def test_exit_equal_values():
    result = converted("year,value\n2022,1200\n2023,1200\n", "exit")
    assert len(result) == 24
    assert result["value"].eq(1200).all()


def test_exit_first_year_flat():
    result = converted("year,value\n2022,1440\n", "exit")
    assert len(result) == 12
    assert result["value"].eq(1440).all()


@pytest.mark.parametrize("mode", ["average", "exit"])
def test_multiple_columns_and_header_normalization(mode):
    result = converted(" Sales , YEAR ,prevalence\n1200,2022,240\n1440,2023,120\n", mode)
    assert result.columns.tolist() == ["year", "month", "Sales", "prevalence"]
    if mode == "average":
        assert result["Sales"].tolist() == [100] * 12 + [120] * 12
        assert result["prevalence"].tolist() == [20] * 12 + [10] * 12
    else:
        assert result["Sales"].tolist() == [1200] * 12 + list(range(1220, 1441, 20))
        assert result["prevalence"].tolist() == [240] * 12 + list(range(230, 119, -10))


@pytest.mark.parametrize("mode", ["average", "exit"])
@pytest.mark.parametrize(
    "content,filename,detail",
    [
        ("year,value\n2022,1200", "yearly.txt", ".csv"),
        (b"year,value\n2022,\xff", "yearly.csv", "decoded"),
        ("value\n1200", "yearly.csv", "year"),
        ("year\n2022", "yearly.csv", "value column"),
        ("year,value\n2022,hello", "yearly.csv", "numeric"),
        ("year,value\n2022,", "yearly.csv", "numeric"),
        ("year,value\n2022,   ", "yearly.csv", "numeric"),
        ("year,value\n,1200", "yearly.csv", "integer"),
        ("year,value\nhello,1200", "yearly.csv", "integer"),
        ("year,value\n2022.5,1200", "yearly.csv", "integer"),
        ("year,value\n2022,1200\n2022,1440", "yearly.csv", "Duplicate years"),
        ("year,value\n2024,1440\n2022,1200", "yearly.csv", "consecutive"),
        ("", "yearly.csv", "data row"),
        (" \n\t", "yearly.csv", "data row"),
        ("year,value\n", "yearly.csv", "data row"),
        ('year,value\n2022,"1200', "yearly.csv", "parsed"),
        ("year,value\n2022,1200,5", "yearly.csv", "same number"),
        ("year,value\n2022", "yearly.csv", "same number"),
        ("year,value\n2022,NaN", "yearly.csv", "numeric"),
        ("year,value\n2022,inf", "yearly.csv", "finite"),
        ("year,value\n2022,-inf", "yearly.csv", "finite"),
        ("year,value\nNaN,1200", "yearly.csv", "integer"),
        ("year,value\ninf,1200", "yearly.csv", "integer"),
        ("year,value, VALUE \n2022,1,2", "yearly.csv", "Duplicate column"),
        ("year, YEAR ,value\n2022,2022,1", "yearly.csv", "Duplicate column"),
        ("year, \n2022,1", "yearly.csv", "names"),
        ("year,value,other\n2022,1,no", "yearly.csv", "other"),
        ("year,value\n2022,1\x00", "yearly.csv", "null"),
    ],
)
def test_validation_errors(content, filename, detail, mode):
    response = upload(content, mode, filename)
    assert response.status_code == 400
    assert detail.casefold() in response.json()["detail"].casefold()


def test_unreadable_upload():
    class UnreadableFile(BytesIO):
        def read(self, *args, **kwargs):
            raise OSError("read failed")

    with pytest.raises(HTTPException) as error:
        convert(UploadFile(filename="yearly.csv", file=UnreadableFile()), Mode.average)
    assert error.value.status_code == 400
    assert "could not be read" in error.value.detail


def test_utf8_bom_and_uppercase_extension():
    response = upload(b"\xef\xbb\xbfyear,value\r\n2022,1200\r\n", filename="YEARLY.CSV")
    assert response.status_code == 200
    assert len(pd.read_csv(StringIO(response.text))) == 12


def test_pure_function_does_not_mutate_input():
    source = pd.DataFrame({" Year ": [2023, 2022], "value": [1440, 1200]})
    original = source.copy(deep=True)
    result = yearly_to_monthly(source, Mode.exit)
    pd.testing.assert_frame_equal(source, original)
    assert result.iloc[12]["value"] == 1220


def test_pure_function_validation():
    with pytest.raises(ValueError, match="consecutive"):
        yearly_to_monthly(pd.DataFrame({"year": [2022, 2024], "value": [1, 2]}), "exit")


def test_no_output_rounding():
    result = converted("year,value\n2022,1\n")
    assert result.iloc[0]["value"] == pytest.approx(1 / 12, rel=1e-15)
    response = upload("year,value\n2022,1\n")
    assert repr(1 / 12) in response.text


def test_exit_decimal_december_matches_exactly():
    source = pd.DataFrame({"year": [2022, 2023], "value": [0.7, 0.1234567890123456]})
    result = yearly_to_monthly(source, "exit")
    assert result.iloc[-1]["value"] == source.iloc[-1]["value"]


@pytest.mark.parametrize("mode", ["invalid", "AVERAGE", ""])
def test_invalid_mode(mode):
    assert upload("year,value\n2022,1200\n", mode).status_code == 422


def test_required_parameters():
    assert client.post("/convert?mode=average").status_code == 422
    assert client.post(
        "/convert", files={"file": ("yearly.csv", "year,value\n2022,1")}
    ).status_code == 422


def test_docs_and_enum():
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["components"]["schemas"]["Mode"]["enum"] == ["average", "exit"]
    operation = schema["paths"]["/convert"]["post"]
    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert "text/csv" in operation["responses"]["200"]["content"]

def test_month_named_value_column_is_preserved():
    result = yearly_to_monthly(pd.DataFrame({"year": [2022], "month": [1200]}), "average")
    assert result.columns.tolist() == ["year", "month", "month"]
    assert result.iloc[:, 1].tolist() == list(range(1, 13))
    assert result.iloc[:, 2].eq(100).all()

