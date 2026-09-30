# Yearly-to-Monthly Percentage API

FastAPI converts each yearly percentage row into 12 monthly rows and returns CSV, JSON, or a PDF with a trend chart and table.

Live Swagger UI: https://yearly-to-monthly-api.onrender.com/docs

## Run locally (Python 3.10+)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

On macOS/Linux, activate with `source .venv/bin/activate`.
Open http://127.0.0.1:8000/docs and use **POST /convert**.
Upload the CSV and select **mode** and **format**.

## Input and calculations

```csv
year,value
2022,6%
2023,8%
2024,12%
```

An optional trailing percent sign is accepted in every value column. `6` and `6%` both mean six percent, while `0.06` means 0.06 percent. Whitespace around the number and percent sign is ignored.

- **average:** minimize squared monthly second differences plus squared excursions outside each column's own yearly minimum and maximum. `RANGE_PENALTY_WEIGHT = 0.1` in smoothing.py controls resistance to excursions: increasing it favors staying near the input range; decreasing it favors smoothness. Yearly means remain hard equality constraints. There are no hard range bounds and no percentage-specific restrictions.
- **exit:** the first year is flat. In later years, month m equals `previous + ((current - previous) / 12) * m`. December is assigned the current yearly percentage exactly. Equal values stay flat; decreasing values produce a straight downward line.

For exit mode, January 2023 is 6.17%, June is 7%, and December is 8%. January 2024 is 8.33% and December is 12%.
Calculations keep full floating-point precision. Only output formatting rounds to two decimal places, removing .00 from whole numbers: `6%`, `6.10%`, `8.17%`.
In exit mode, small differences between rounded displayed monthly steps do not change the underlying linear calculation. In average mode, the mean constraint holds on unrounded values within floating-point tolerance; the displayed two-decimal values can have a slightly different mean.

Average's first month has a hard positive lower limit: `epsilon = max(1e-6, 0.01 * max(data_max, 0))`, independently per value column. This is a first-month anchor only, not a range constraint for the entire curve. The absolute floor handles zero and negative maxima. Positive constants above epsilon stay flat; zero, negative, or extremely small constant targets require optimization to preserve their means while starting positive. Later months can be negative. Scaling invariance holds when the absolute floor is inactive; it deliberately does not hold below that floor.

A finite soft penalty does **not** guarantee strict containment or restrict all overshoot to sharp jumps. For example, 6/8/12 gives roughly 5.97-12.21 at the default weight. Strict containment plus an exact mean at the minimum/maximum would require those whole years to be flat. The implemented objective is the requested soft tradeoff, not a hidden hard bound or jump detector. Raw first-month values are positive; unchanged two-decimal percentage formatting may show very small positive values as `0%`. Unrounded values are available from the pure conversion function.

Each column is normalized for solver stability. Sparse operators and explicit nonnegative hinge slack variables avoid dense matrices and redundant solver variables. Small equality residuals are removed numerically; if first-month roundoff is repaired, the other eleven months are compensated to preserve the year's mean. Non-finite outputs and failed convergence still produce errors. This is not a reproduction of a Matson-Jack formula.

For the review example, set Y10=98 and append Y11-Y13=100 in the input. The API never alters or appends source targets. This deliberately lowers Moderate/Fast from Y9 99/99.8 to Y10 98.

Multiple percentage columns are processed independently. Headers are trimmed and compared case-insensitively; value names retain their trimmed spelling and original order. Output begins with integer `year` and `month` columns, sorted by year then month.

## Output formats

The required `mode` query parameter is `average` or `exit`.
The optional `format` query parameter is `csv` (default), `json`, or `pdf`.

| Format | Content type | Response |
| --- | --- | --- |
| csv | text/csv | Download named monthly_<mode>.csv |
| json | application/json | Array of monthly objects directly in the body |
| pdf | application/pdf | Download named monthly_<mode>.pdf |

All three formats show the same percentage strings. JSON keeps year and month as integers. Example for a single-year 6% input (or the first year in exit mode):

```json
[
  {"year": 2022, "month": 1, "value": "6%"},
  {"year": 2022, "month": 2, "value": "6%"}
]
```

PDF reports include a continuous line for each value column and a table containing all monthly rows. Chart points use unrounded data and equally spaced monthly positions across years. Long tables continue across pages with repeated headers; wide tables are split into column groups.

CSV/PDF preserve a value column named `month` after the generated month column. Because JSON object keys must be unique, JSON aliases that value column to `month_value`, appending another `_value` if that name already exists.

## Example requests

Use the included `yearly.csv`. On Windows PowerShell, use `curl.exe`.

```sh
curl -X POST "http://127.0.0.1:8000/convert?mode=average" -F "file=@yearly.csv" -o monthly_average.csv
curl -X POST "http://127.0.0.1:8000/convert?mode=exit&format=json" -F "file=@yearly.csv"
curl -X POST "http://127.0.0.1:8000/convert?mode=exit&format=pdf" -F "file=@yearly.csv" -o monthly_exit.pdf
```

Replace the local base URL with `https://yearly-to-monthly-api.onrender.com` to use the hosted API.
GET /convert is also supported with the same multipart file body and query parameters. Use POST in Swagger and browser clients, which generally cannot send file bodies with GET. Opening the conversion URL alone does not supply a file.

## Validation

Invalid input returns HTTP 400 with a clear JSON `detail`, regardless of output format:

- Wrong file extension, unreadable content, invalid UTF-8, malformed CSV, or null characters.
- Missing year column, missing value columns, empty or duplicate headers, or no data rows.
- Empty, nonnumeric, NaN, or infinite values (including malformed percentage strings such as abc% or 6%%).
- Non-integer years, duplicate years, or gaps between years after sorting.

UTF-8 BOMs and uppercase .CSV extensions are supported. Finite absolute values and percentages are accepted without a 0-100 restriction. Exit calculations are unchanged.
Missing required request parameters or invalid mode/format values return HTTP 422.

## Tests and code

```sh
python -m pytest -q
```

- `main.yearly_to_monthly(df, mode)`: pure conversion and validation; returns a numeric DataFrame without modifying the input.
- `outputs.formatted_monthly(df)`: shared percentage formatting.
- `outputs.to_csv_response`, `to_json_response`, `to_pdf_response`: format-specific response builders.
- `outputs.build_pdf(df, mode)`: independently testable in-memory PDF generation.
- `outputs.build_trend_chart(df)`: plots all monthly numeric values without smoothing, randomization, or rounding.

Tests cover the existing validation cases, both calculation modes, multiple columns, backward-compatible numeric inputs, matching CSV/JSON/PDF values, chart linearity, PDF pagination, download headers, and Swagger dropdowns. pypdf is used to inspect PDF table text and embedded charts in tests.

## Render deployment

The public repository is https://github.com/ujwalreddybattu04/yearly-to-monthly-api.
The included render.yaml defines a free Python web service. The .python-version file selects Python 3.12.

- Build: `python -m pip install -r requirements.txt && python -m pytest -q`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/docs`

For a new deployment, connect the repository through Render's **New > Blueprint** flow and deploy render.yaml. Keep main.py, outputs.py, requirements.txt, render.yaml, and .python-version at the repository root, with tests/ beside them.

The service runs independently of your computer. The free instance sleeps after 15 idle minutes and automatically wakes on the next request, which can take about a minute.
See https://render.com/docs/deploy-fastapi and https://render.com/docs/free.

## Robustness and Unicode fonts

Every calculated monthly value is checked for finiteness before formatting.
Arithmetic overflow returns HTTP 400 naming the affected column.
Average scales each column internally for numerical stability and rejects non-finite calculated results; Exit retains its existing behavior.

Chart text is literal: dollar signs and backslashes are not interpreted as
LaTeX or mathematical expressions. Charts and PDF tables both use the bundled
API Unicode Sans font, derived from Noto Sans CJK SC. It includes Chinese,
Japanese, Korean, Latin, Greek, and Cyrillic glyphs from the source font.
See fonts/README.md and fonts/LICENSE.txt for provenance and licensing.
The font is bundled with the application; no server-installed font is required.
Keep the fonts/ directory when copying or deploying this project.

Value cells still require numeric percentages. Non-Latin text is supported in
headers; a nonnumeric value such as å…­% is still invalid. Coverage is limited
to the bundled font's repertoire, not every Unicode script or emoji.

For characters absent from the CJK font, both renderers use Matplotlib's bundled DejaVu Sans as a fallback, including accented Greek. PDF font runs are escaped as literal text before rendering.

All API outputs retain percentage formatting. The query parameters are `mode` and `format`; there is no units selector.
