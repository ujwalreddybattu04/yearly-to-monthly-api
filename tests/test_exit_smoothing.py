"""Exit is one global minimum-curvature curve with December anchors."""
from decimal import Decimal, localcontext
from io import BytesIO, StringIO
from time import perf_counter

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from main import app, yearly_to_monthly
from smoothing import smooth_exit, _verify_december_values


def solve(targets):
    return smooth_exit(np.asarray(targets, dtype=float).reshape(-1, 1), ["value"])[:, 0]


def assert_decembers(values, targets):
    with localcontext() as context:
        context.prec = 800
        for actual, target in zip(values[11::12], targets):
            assert Decimal(str(actual)) == Decimal(str(float(target)))


def test_two_year_solution_is_one_line_including_first_year():
    values = solve([6, 8])
    # With two anchors and an inactive guard, a straight line has zero curvature.
    expected = 6 + (np.arange(1, 25) - 12) * (2 / 12)
    np.testing.assert_allclose(values, expected, atol=1e-8, rtol=0)
    assert np.ptp(values[:12]) > 1
    assert values[0] < 6
    assert_decembers(values, [6, 8])


@pytest.mark.parametrize("targets,active_guard", [([6, 8, 12], False), ([0.01, 80, 100], True)])
def test_matches_independent_minimum_curvature_equations(targets, active_guard):
    # Solve the small equality-constrained quadratic system independently of CVXPY.
    months = 12 * len(targets)
    differences = np.diff(np.eye(months), n=2, axis=0)
    anchors = np.eye(months)[11::12]
    rhs = np.array(targets, dtype=float)
    if active_guard:
        anchors = np.vstack((anchors, np.eye(months)[0]))
        rhs = np.append(rhs, 0)
    kkt = np.block([[differences.T @ differences, anchors.T],
                    [anchors, np.zeros((len(rhs), len(rhs)))]] )
    expected = np.linalg.solve(kkt, np.concatenate((np.zeros(months), rhs)))[:months]
    actual = solve(targets)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    assert actual[0] >= 0
    assert_decembers(actual, targets)


def test_curvature_spans_december_boundaries():
    values = solve([6, 8, 12, 9, 10])
    curvature = np.abs(np.diff(values, n=2))
    # Each row of the second difference is centered on month index row + 1.
    boundary_rows = np.array([11, 23, 35, 47]) - 1
    inside_year = np.delete(curvature, boundary_rows)
    assert curvature[boundary_rows].max() <= 2 * inside_year.max() + 1e-8


@pytest.mark.parametrize("targets", [[0.01, 80, 100], [0, 100], [-6, -8], [-2, 4, -1]])
def test_first_month_guard_and_exact_decembers(targets):
    values = solve(targets)
    assert np.isfinite(values).all()
    assert values[0] >= 0
    assert_decembers(values, targets)


@pytest.mark.parametrize("value", [0, 6, 1e-100, 1e308])
def test_single_year_flat_fallback(value):
    np.testing.assert_array_equal(solve([value]), np.full(12, value))


def test_negative_single_year_conflict_is_clear_400():
    response = TestClient(app).post('/convert?mode=exit',
        files={'file': ('yearly.csv', 'year,value\n2022,-6')})
    assert response.status_code == 400
    assert "negative single-year target" in response.json()['detail']


def test_extreme_ratio_rejected_before_solver(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('Unsafe range must be rejected before constructing the optimizer')
    monkeypatch.setattr('smoothing.cp.Problem', fail)
    with pytest.raises(ValueError, match="Column 'value'.*too wide a range"):
        solve([1, 1e308])


@pytest.mark.parametrize("fmt", ['csv', 'json', 'pdf'])
def test_extreme_ratio_returns_400_for_all_outputs(fmt):
    response = TestClient(app).post('/convert', params={'mode':'exit', 'format':fmt},
        files={'file': ('yearly.csv', 'year,value\n1,1\n2,1e308')})
    assert response.status_code == 400
    assert "too wide a range" in response.json()['detail']


@pytest.mark.parametrize("targets", [[1, 100000], [1, 1e12], [1e-100, 2e-100], [1e308, 1e308]])
def test_safe_scales_still_work(targets):
    values = solve(targets)
    assert np.isfinite(values).all()
    assert values[0] >= 0
    assert_decembers(values, targets)


def test_actual_overflow_is_rejected():
    with pytest.raises(ValueError, match='non-finite result'):
        solve([-1.79e308, 1.79e308])


def test_decimal_verifier_detects_corrupted_december_and_nonfinite_values():
    values = np.ones(24)
    values[11] = 2
    with pytest.raises(ValueError, match='December accuracy check for year index 0'):
        _verify_december_values(values, np.ones(2), 'value')
    values[11] = 1
    values[4] = np.inf
    with pytest.raises(ValueError, match='non-finite result'):
        _verify_december_values(values, np.ones(2), 'value')


@pytest.mark.parametrize('targets', [[[6]], [[6], [6]], [[6], [8]]])
def test_decimal_verification_runs_on_all_return_paths(monkeypatch, targets):
    import smoothing
    calls = []
    original = smoothing._verify_december_values
    def record(values, expected, name):
        calls.append(name)
        original(values, expected, name)
    monkeypatch.setattr(smoothing, '_verify_december_values', record)
    smooth_exit(np.array(targets, dtype=float), ['value'])
    assert calls == ['value']


def test_fifty_years_five_columns_under_one_second():
    targets = np.linspace(1, 100, 50)[:, None] * np.arange(1, 6)[None, :]
    started = perf_counter()
    result = smooth_exit(targets)
    elapsed = perf_counter() - started
    assert elapsed < 1, f'Exit 50 years x 5 columns took {elapsed:.3f}s'
    np.testing.assert_array_equal(result[11::12], targets)


def test_new_exit_csv_json_pdf_share_the_same_monthly_values():
    client = TestClient(app)
    source = 'year,value,other\n2022,6%,12%\n2023,8%,8%\n2024,12%,6%'
    responses = {fmt: client.post('/convert', params={'mode':'exit', 'format':fmt},
                 files={'file': ('yearly.csv', source)}) for fmt in ('csv','json','pdf')}
    assert all(r.status_code == 200 for r in responses.values())
    rows = pd.read_csv(StringIO(responses['csv'].text))
    assert rows.to_dict('records') == responses['json'].json()
    pdf = responses['pdf']
    assert pdf.headers['content-type'] == 'application/pdf'
    assert pdf.content.startswith(b'%PDF-')
    reader = PdfReader(BytesIO(pdf.content))
    text = '\n'.join(page.extract_text() for page in reader.pages)
    assert all(value in text for column in ('value', 'other') for value in rows[column])
    assert rows.loc[:11, 'value'].nunique() > 1
    assert rows.loc[[11, 23, 35], 'value'].tolist() == ['6%', '8%', '12%']


def test_main_uses_exit_smoother_once_for_all_years_and_columns(monkeypatch):
    from unittest.mock import Mock
    wrapped = Mock(wraps=smooth_exit)
    monkeypatch.setattr('main.smooth_exit', wrapped)
    frame = pd.DataFrame({'year': [2023, 2022], 'value': [8., 6.], 'other': [4., 2.]})
    result = yearly_to_monthly(frame, 'exit')
    wrapped.assert_called_once()
    np.testing.assert_array_equal(wrapped.call_args.args[0], [[6, 2], [8, 4]])
    assert result.year.tolist() == [2022] * 12 + [2023] * 12
