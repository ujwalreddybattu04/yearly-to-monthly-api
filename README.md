# Yearly-to-Monthly CSV API

A Python 3.10+ FastAPI application that expands each yearly CSV row into 12 monthly rows.

## Install and run

From this directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

Open http://127.0.0.1:8000/docs, expand **POST /convert**, click **Try it out**, select a mode, upload a CSV, and execute. The response is a CSV attachment named `monthly_average.csv` or `monthly_exit.csv`.

## Example request

Save this as `yearly.csv`:

```csv
year,value
2022,1200
2023,1440
```

```sh
curl -X POST "http://127.0.0.1:8000/convert?mode=exit" -F "file=@yearly.csv" -o monthly_exit.csv
```

In Windows PowerShell, use `curl.exe` instead of `curl`. Change the query to `mode=average` and the output filename to `monthly_average.csv` for totals.

- **average** divides each value by 12. The example produces twelve rows of 100 for 2022 and twelve rows of 120 for 2023.
- **exit** treats values as year-end levels. All 2022 months equal 1200. In 2023, values are 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380, 1400, 1420, and 1440.
- The first year in exit mode is flat. Later years interpolate from the previous December; December is assigned the current yearly value exactly.

Each numeric column is converted independently. Output columns are `year,month`, followed by input value columns in their original order. Rows are sorted by year and month.

## Input and errors

Uploads must have a `.csv` extension (case-insensitive) and UTF-8 encoding; a UTF-8 BOM is supported. Headers are trimmed and compared case-insensitively. Value-column spelling is preserved after trimming; the year column becomes `year`.

A CSV requires an integer-valued `year` column, at least one numeric value column, and at least one data row. Years must be unique and consecutive after sorting. Empty, nonnumeric, NaN, or infinite values are rejected. Malformed CSV records and duplicate or empty headers are also rejected. An input value column named `month` is preserved after the generated month column, so that case produces two output columns with the same name.

Invalid CSV inputs return HTTP 400 with a descriptive JSON `detail`. Missing request parameters or modes outside `average` and `exit` return FastAPI's standard HTTP 422 validation response.

Values use standard double-precision floating-point arithmetic. CSV output applies no explicit rounding or fixed decimal formatting. Non-terminating fractions such as 1/12 have normal floating-point representation limits, so sums may differ from the original total by floating-point error.

## Tests and reusable conversion

```sh
python -m pytest -q
```

`main.yearly_to_monthly(df, mode)` accepts a pandas DataFrame and either a `Mode` member or its string value, returns a new DataFrame, and raises `ValueError` for invalid data. It does not modify the input and has no dependency on request objects.

The tests cover both modes, multiple columns, sorting, input validation, download headers, precision, the pure function, and the OpenAPI mode dropdown.

## Deploy publicly on Render

The included render.yaml defines a Python web service on Render's free plan.
The .python-version file selects the latest Python 3.12 patch release.
Every build installs requirements and runs the tests before starting the API.

1. Create a repository at https://github.com/new (a private repository works).
2. Upload the contents of this project to the repository. Put main.py,
   requirements.txt, render.yaml, and .python-version at the repository root,
   with tests/test_convert.py inside tests/. Include the hidden .python-version
   and .gitignore files. Do not upload the ZIP itself, .venv, or log files.
3. Sign in at https://dashboard.render.com and select New > Blueprint.
4. Connect your GitHub account and select the repository.
5. Render reads render.yaml. Review the service and confirm the free plan,
   then select Deploy Blueprint.
6. When the service is Live, open its assigned HTTPS URL with /docs appended.
   Share that URL so others can upload CSV files and download monthly results.

For manual setup with New > Web Service, choose Python 3 and use:

- Build command: python -m pip install -r requirements.txt && python -m pytest -q
- Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
- Instance type: Free
- Health check path: /docs
- Root directory: leave empty when project files are at the repository root.

Example endpoint after deployment (replace YOUR-SERVICE with Render's assigned name):

    https://YOUR-SERVICE.onrender.com/convert?mode=average

The service runs independently of your computer. Render's free service sleeps
after 15 idle minutes and wakes when requested; the first request can take
about a minute. An always-running service requires a paid instance.
The API is public: anyone with the URL can submit a CSV.

Official references:
- https://render.com/docs/deploy-fastapi
- https://render.com/docs/infrastructure-as-code
- https://render.com/docs/python-version
- https://render.com/docs/free
