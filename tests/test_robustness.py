"""Regression coverage for finite results and literal Unicode PDF text."""
from io import BytesIO
import warnings

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from matplotlib.ft2font import FT2Font
from matplotlib.text import Text
from pypdf import PdfReader

from main import app, yearly_to_monthly, _checked_monthly_frame
from outputs import FONT_PATH, FALLBACK_FONT_PATH, CHART_FONT_FAMILIES, build_trend_chart

client = TestClient(app)


def upload(content, mode="average", fmt="pdf"):
    return client.post(
        "/convert", params={"mode": mode, "format": fmt},
        files={"file": ("yearly.csv", content.encode("utf-8"), "text/csv")},
    )


@pytest.mark.parametrize("fmt", ["csv", "json", "pdf"])
@pytest.mark.parametrize("extreme", ["1e308%", "-1e308%"])
def test_large_finite_constants_remain_accepted(fmt, extreme):
    response = upload(f"year,large\n2022,{extreme}\n2023,{extreme}", fmt=fmt)
    assert response.status_code == 200


@pytest.mark.parametrize("fmt", ["csv", "json", "pdf"])
def test_real_overflow_returns_400(fmt):
    response = upload("year,large\n1,-1e308\n2,1e308", fmt=fmt)
    assert response.status_code == 400
    assert "non-finite result" in response.json()["detail"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_result_validator_checks_value_columns_by_position(value):
    with pytest.raises(ValueError, match="Column 'month' produced a non-finite result"):
        _checked_monthly_frame([[2022, 1, 6.0, value]], ["normal", "month"])


@pytest.mark.parametrize("mode,years", [("average", 1), ("average", 2), ("exit", 1), ("exit", 2)])
def test_all_conversion_paths_validate_results(monkeypatch, mode, years):
    from unittest.mock import Mock
    check = Mock(wraps=_checked_monthly_frame)
    monkeypatch.setattr("main._checked_monthly_frame", check)
    result = yearly_to_monthly(
        pd.DataFrame({"year": range(2022, 2022 + years), "value": [6.0] * years}), mode
    )
    check.assert_called_once()
    assert len(result) == 12 * years


def test_finite_large_exit_values_remain_accepted():
    response = upload("year,value\n2022,1e308%\n2023,1e308%", mode="exit", fmt="json")
    assert response.status_code == 200
    assert len(response.json()) == 24
    assert all(np.isfinite(float(row["value"].removesuffix("%"))) for row in response.json())


@pytest.mark.parametrize("header", [
    "$notacommand$", r"$\notacommand$", r"$\frac{broken$",
    r"$x_1^{2}$", "sales & <value> #50% {test}", "\u60a3\u75c5\u7387_\u65e5\u672c\u8a9e_\ud55c\uad6d\uc5b4",
])
@pytest.mark.parametrize("mode", ["average", "exit"])
def test_special_headers_are_literal_in_pdf(header, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = upload(f"year,{header}\n2022,6%\n2023,8%", mode)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    reader = PdfReader(BytesIO(response.content))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert header in text
    assert not any("missing from font" in str(w.message) for w in caught)


def test_chart_disables_math_for_all_text_and_preserves_header():
    header = "\u60a3\u75c5\u7387 $\\notacommand$"
    data = yearly_to_monthly(pd.DataFrame({"year": [2022, 2023], header: [6, 8]}), "average")
    figure = build_trend_chart(data)
    try:
        figure.canvas.draw()
        assert [t.get_text() for t in figure.axes[0].get_legend().get_texts()] == [header]
        for text in figure.findobj(Text):
            assert text.get_parse_math() is False
            assert text.get_usetex() is False
            assert text.get_fontproperties().get_family() == CHART_FONT_FAMILIES
    finally:
        figure.clear()


def test_cjk_font_has_glyphs_and_is_embedded_in_pdf():
    label = "\u60a3\u75c5\u7387_\u65e5\u672c\u8a9e_\ud55c\uad6d\uc5b4_\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac_\u041a\u0438\u0440\u0438\u043b\u043b\u0438\u0446\u0430"
    font = FT2Font(str(FONT_PATH))
    fallback = FT2Font(str(FALLBACK_FONT_PATH))
    assert all(font.get_char_index(ord(character)) or fallback.get_char_index(ord(character)) for character in label)
    response = upload(f"year,{label}\n2022,6%")
    assert response.status_code == 200
    reader = PdfReader(BytesIO(response.content))
    assert label in "\n".join(page.extract_text() for page in reader.pages)
    embedded = []
    for page in reader.pages:
        for reference in page["/Resources"]["/Font"].values():
            pdf_font = reference.get_object()
            descriptor = pdf_font.get("/FontDescriptor")
            if descriptor:
                descriptor = descriptor.get_object()
                if "/FontFile2" in descriptor:
                    embedded.append(descriptor["/FontFile2"].get_object().get_data())
    assert embedded and all(len(data) > 1000 for data in embedded)


def test_chinese_nonnumeric_values_still_rejected():
    response = upload("year,value\n2022,\u516d%")
    assert response.status_code == 400
    assert "numeric" in response.json()["detail"]
