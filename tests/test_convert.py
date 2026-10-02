from io import BytesIO, StringIO

import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from main import Mode, app, convert, yearly_to_monthly


client = TestClient(app)


def upload(content, mode="average", filename="yearly.csv", output_format=None):
    return client.post(
        "/convert",
        params={"mode": mode, **({"format": output_format} if output_format is not None else {})},
        files={"file": (filename, content, "text/csv")},
    )


def converted(content, mode="average"):
    response = upload(content, mode)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == (
        f'attachment; filename="monthly_{mode}.csv"'
    )
    frame = pd.read_csv(StringIO(response.text))
    for column in frame.columns[2:]:
        frame[column] = frame[column].str.removesuffix("%").astype(float)
    return frame


def test_average_single_year():
    result = converted("year,value\n2022,12\n")
    assert len(result) == 12
    assert result["month"].tolist() == list(range(1, 13))
    assert result["year"].eq(2022).all()
    assert result["value"].eq(12).all()
    assert result["value"].mean() == 12


def test_average_two_years_sorted():
    result = converted("year,value\n2023,24\n2022,12\n")
    assert len(result) == 24
    assert result["year"].tolist() == [2022] * 12 + [2023] * 12
    assert result["month"].tolist() == list(range(1, 13)) * 2
    assert result.groupby("year")["value"].mean().tolist() == pytest.approx([12,24], abs=0.01)
    assert result["value"].nunique() > 2


def test_exit_example():
    result = converted("year,value\n2023,1440\n2022,1200\n", "exit")
    assert len(result) == 24
    assert result["value"].tolist() == pytest.approx(list(range(980, 1441, 20)))


def test_exit_equal_values():
    result = converted("year,value\n2022,1200\n2023,1200\n", "exit")
    assert len(result) == 24
    assert result["value"].eq(1200).all()


def test_exit_single_year_flat():
    result = converted("year,value\n2022,1440\n", "exit")
    assert len(result) == 12
    assert result["value"].eq(1440).all()


@pytest.mark.parametrize("mode", ["average", "exit"])
def test_multiple_columns_and_header_normalization(mode):
    content = " Sales , YEAR ,prevalence\n1200,2022,240\n1440,2023,120\n"
    if mode == "average":
        content = " Sales , YEAR ,prevalence\n12,2022,24\n24,2023,12\n"
    result = converted(content, mode)
    assert result.columns.tolist() == ["year", "month", "Sales", "prevalence"]
    if mode == "average":
        assert result.groupby("year")["Sales"].mean().tolist() == pytest.approx([12,24], abs=0.01)
        assert result.groupby("year")["prevalence"].mean().tolist() == pytest.approx([24,12], abs=0.01)
    else:
        assert result["Sales"].tolist() == pytest.approx(list(range(980, 1441, 20)))
        assert result["prevalence"].tolist() == pytest.approx(list(range(350, 119, -10)))


@pytest.mark.parametrize("mode", ["average", "exit"])
@pytest.mark.parametrize(
    "content,filename,detail",
    [
        ("year,value\n2022,1200", "yearly.txt", ".csv"),
        (b"year,value\n2022,\x81", "yearly.csv", "decoded"),
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
    response = upload(b"\xef\xbb\xbfyear,value\r\n2022,12\r\n", filename="YEARLY.CSV")
    assert response.status_code == 200
    assert len(pd.read_csv(StringIO(response.text))) == 12


@pytest.mark.parametrize("mode", ["average", "exit"])
def test_windows_1252_csv_upload(mode):
    # Excel's Windows CSV uses byte 0x96 for the dash in these real headers.
    source = (
        b"year,10-Year \x96 Slow,10-Year \x96 Fast\r\n"
        b"1,1.00%,8.00%\r\n"
        b"2,4.00%,25.00%\r\n"
    )
    response = upload(source, mode)
    assert response.status_code == 200, response.text
    result = pd.read_csv(StringIO(response.text))
    assert len(result) == 24
    assert result.columns.tolist() == [
        "year", "month", "10-Year \u2013 Slow", "10-Year \u2013 Fast"
    ]


def test_pure_function_does_not_mutate_input():
    source = pd.DataFrame({" Year ": [2023, 2022], "value": [1440, 1200]})
    original = source.copy(deep=True)
    result = yearly_to_monthly(source, Mode.exit)
    pd.testing.assert_frame_equal(source, original)
    assert result.iloc[12]["value"] == pytest.approx(1220)


def test_pure_function_validation():
    with pytest.raises(ValueError, match="consecutive"):
        yearly_to_monthly(pd.DataFrame({"year": [2022, 2024], "value": [1, 2]}), "exit")


def test_round_only_when_formatting_output():
    source = pd.DataFrame({"year": [2022, 2023], "value": [6, 8]})
    result = yearly_to_monthly(source, "exit")
    assert result.iloc[12]["value"] == pytest.approx(6 + 2 / 12, abs=1e-8)
    assert result.iloc[12]["value"] != round(result.iloc[12]["value"], 2)
    response = upload("year,value\n2022,6%\n2023,8%", "exit")
    assert "2023,1,6.17%" in response.text


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
    assert set(operation["responses"]["200"]["content"]) == {"text/csv", "application/json", "application/pdf"}
    assert schema["components"]["schemas"]["OutputFormat"]["enum"] == ["csv", "json", "pdf"]
    output_parameter = next(p for p in operation["parameters"] if p["name"] == "format")
    assert output_parameter["schema"]["default"] == "csv"

def test_month_named_value_column_is_preserved():
    result = yearly_to_monthly(pd.DataFrame({"year": [2022], "month": [12]}), "average")
    assert result.columns.tolist() == ["year", "month", "month"]
    assert result.iloc[:, 1].tolist() == list(range(1, 13))
    assert result.iloc[:, 2].eq(12).all()


@pytest.mark.parametrize("mode", ["average", "exit"])
def test_percentage_strings_and_plain_numbers_match(mode):
    signed = upload("year,value\n2022,6%\n2023,8%", mode)
    plain = upload("year,value\n2022,6\n2023,8", mode)
    assert signed.status_code == plain.status_code == 200
    assert signed.content == plain.content


def test_average_percentage_level():
    result = upload("year,value\n2022,6%")
    rows = pd.read_csv(StringIO(result.text))
    assert rows["value"].tolist() == ["6%"] * 12


def test_exit_percentage_endpoints():
    result = upload("year,value\n2022,6%\n2023,8%", "exit")
    rows = pd.read_csv(StringIO(result.text))
    assert rows.iloc[0]["value"] == "4.17%"
    assert rows.iloc[11]["value"] == "6%"
    assert rows.iloc[12]["value"] == "6.17%"
    assert rows.iloc[-1]["value"] == "8%"


@pytest.mark.parametrize("bad", ["abc%", "%", "6%%", "%6", "NaN%", "inf%", "-inf%"])
@pytest.mark.parametrize("output_format", ["csv", "json", "pdf"])
def test_invalid_percentages(bad, output_format):
    response = upload(f"year,value\n2022,{bad}", output_format=output_format)
    assert response.status_code == 400
    assert "numeric" in response.json()["detail"]


@pytest.mark.parametrize("mode", ["average", "exit"])
def test_json_matches_csv(mode):
    content = "year,value,other\n2024,12%,4%\n2022,6%,8%\n2023,8%,6%"
    csv_response = upload(content, mode)
    response = upload(content, mode, output_format="json")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert "content-disposition" not in response.headers
    assert response.json() == pd.read_csv(StringIO(csv_response.text)).to_dict("records")
    assert len(response.json()) == 36
    assert isinstance(response.json()[0]["year"], int)
    assert isinstance(response.json()[0]["month"], int)


@pytest.mark.parametrize("mode", ["average", "exit"])
def test_pdf_contains_chart_and_matching_table(mode):
    from pypdf import PdfReader
    content = "year,value\n2022,6%\n2023,8%\n2024,12%"
    response = upload(content, mode, output_format="pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == f'attachment; filename="monthly_{mode}.pdf"'
    assert response.content.startswith(b"%PDF")
    reader = PdfReader(BytesIO(response.content))
    text = "\n".join(page.extract_text() for page in reader.pages)
    csv_rows = pd.read_csv(StringIO(upload(content, mode).text))
    for value in csv_rows["value"]:
        assert value in text
    assert "2022" in text and "2024" in text
    assert sum(len(page.images) for page in reader.pages) >= 1


def test_pdf_builder_is_independent_and_paginates():
    from outputs import build_pdf
    from pypdf import PdfReader
    source = pd.DataFrame({"year": list(range(2000, 2010)), **{f"metric{i}": [i] * 10 for i in range(8)}})
    data = yearly_to_monthly(source, "average")
    original = data.copy(deep=True)
    reader = PdfReader(BytesIO(build_pdf(data, "average")))
    assert len(reader.pages) > 2
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "metric0" in text and "metric7" in text and "2009" in text
    pd.testing.assert_frame_equal(data, original)


def test_chart_uses_smoothed_unrounded_data():
    from outputs import build_trend_chart
    data = yearly_to_monthly(pd.DataFrame({"year": [2022, 2023, 2024], "value": ["6%", "8%", "12%"], "other": [12, 8, 6]}), "exit")
    figure = build_trend_chart(data)
    try:
        assert len(figure.axes[0].lines) == 2
        for index, line in enumerate(figure.axes[0].lines):
            assert list(line.get_xdata()) == list(range(36))
            assert list(line.get_ydata()) == data.iloc[:, index + 2].tolist()
        values = list(figure.axes[0].lines[0].get_ydata())
        assert [values[11], values[23], values[35]] == [6, 8, 12]
        assert any(value != round(value, 2) for value in values)
    finally:
        figure.clear()


@pytest.mark.parametrize("value,expected", [(6, "6%"), (8.166666, "8.17%"), (12, "12%"), (6.1, "6.10%"), (-0.001, "0%"), (-2.3456, "-2.35%")])
def test_percentage_formatting(value, expected):
    from outputs import format_percentage
    assert format_percentage(value) == expected


@pytest.mark.parametrize("invalid", ["xml", "CSV", ""])
def test_invalid_format(invalid):
    assert upload("year,value\n2022,6%", output_format=invalid).status_code == 422


def test_explicit_csv_equals_default():
    content = "year,value\n2022,6%"
    assert upload(content).content == upload(content, output_format="csv").content


def test_get_with_multipart_upload():
    response = client.request("GET", "/convert?mode=average&format=json", files={"file": ("yearly.csv", "year,value\n2022,6%")})
    assert response.status_code == 200
    assert response.json() == [{"year": 2022, "month": m, "value": "6%"} for m in range(1, 13)]


def test_json_month_collision_preserves_all_values():
    response = upload("year,month,month_value\n2022,6%,8%", output_format="json")
    assert response.status_code == 200
    assert response.json()[0] == {"year": 2022, "month": 1, "month_value_value": "6%", "month_value": "8%"}


def test_whitespace_percentage_input():
    response = upload("year,value\n2022, 6 % ")
    assert response.status_code == 200
    assert pd.read_csv(StringIO(response.text))["value"].eq("6%").all()

@pytest.mark.parametrize("values", [[6, 8, 12], [12, 8, 6], [26, 32, 23, 29], [6, 6, 6], [6, 8]])
def test_average_spline_preserves_each_year_mean(values):
    source = pd.DataFrame({"year": range(2022, 2022 + len(values)), "value": values,
                           "second": [v * 0.3 + 1 for v in values]})
    original = source.copy(deep=True)
    result = yearly_to_monthly(source.iloc[::-1], "average")
    assert result.groupby("year")["value"].mean().tolist() == pytest.approx(values, abs=1e-9, rel=0)
    assert result.groupby("year")["second"].mean().tolist() == pytest.approx(source["second"], abs=1e-9, rel=0)
    pd.testing.assert_frame_equal(source, original)


def test_average_example_varies_and_has_small_boundary_steps():
    data = yearly_to_monthly(pd.DataFrame({"year": [2022, 2023, 2024], "value": ["6%", "8%", "12%"]}), "average")
    assert data.groupby("year")["value"].nunique().gt(1).all()
    values = data["value"].tolist()
    for boundary in (12, 24):
        step = values[boundary] - values[boundary - 1]
        left = values[boundary - 1] - values[boundary - 2]
        right = values[boundary + 1] - values[boundary]
        assert 0 < step < 0.5
        assert step < 2 * max(left, right)


def test_average_single_year_skips_spline(monkeypatch):
    def unexpected_spline(*args, **kwargs):
        pytest.fail("Single year must not construct a spline")
    monkeypatch.setattr("smoothing.cp.Problem", unexpected_spline)
    result = yearly_to_monthly(pd.DataFrame({"year": [2022], "value": ["6%"]}), "average")
    assert result["value"].tolist() == [6.0] * 12


def test_average_constant_years_remain_constant():
    result = yearly_to_monthly(pd.DataFrame({"year": [2022, 2023, 2024], "value": [6, 6, 6]}), "average")
    assert result["value"].tolist() == [6.0] * 36


def test_average_chart_and_all_formats_use_corrected_values():
    from outputs import build_trend_chart, formatted_monthly
    from pypdf import PdfReader
    source = pd.DataFrame({"year": [2022, 2023, 2024], "value": [6, 8, 12]})
    data = yearly_to_monthly(source, "average")
    figure = build_trend_chart(data)
    try:
        assert list(figure.axes[0].lines[0].get_ydata()) == data["value"].tolist()
    finally:
        figure.clear()
    content = "year,value\n2022,6%\n2023,8%\n2024,12%"
    expected = formatted_monthly(data)
    csv_response = upload(content)
    json_response = upload(content, output_format="json")
    pdf_response = upload(content, output_format="pdf")
    assert csv_response.status_code == json_response.status_code == pdf_response.status_code == 200
    assert pd.read_csv(StringIO(csv_response.text)).to_dict("records") == expected.to_dict("records")
    assert json_response.json() == expected.to_dict("records")
    reader = PdfReader(BytesIO(pdf_response.content))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert all(value in text for value in expected["value"])
    assert sum(len(page.images) for page in reader.pages) >= 1
