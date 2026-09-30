"""Yearly-to-monthly CSV API. Run with uvicorn main:app --reload."""

import csv
from decimal import Decimal, InvalidOperation
from enum import Enum
from io import StringIO
import math

import numpy as np
import pandas as pd
from smoothing import smooth_average
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from outputs import to_csv_response, to_json_response, to_pdf_response


class Mode(str, Enum):
    average = "average"
    exit = "exit"


class OutputFormat(str, Enum):
    csv = "csv"
    json = "json"
    pdf = "pdf"


app = FastAPI(title="Yearly-to-Monthly Percentage Converter")


def yearly_to_monthly(df: pd.DataFrame, mode: Mode | str) -> pd.DataFrame:
    """Validate and expand yearly data without mutating df.

    Value headers retain their spelling after trimming. Header comparisons are
    case-insensitive. Raises ValueError for invalid input.
    """
    mode = Mode(mode)
    data = df.copy(deep=True)
    names = [str(column).strip() for column in data.columns]
    keys = [name.casefold() for name in names]
    if "year" not in keys:
        raise ValueError("Missing required 'year' column.")
    if len(names) < 2:
        raise ValueError("At least one numeric value column is required.")
    if any(not name for name in names):
        raise ValueError("Column names must not be empty.")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate column names are not allowed (case-insensitive).")
    names[keys.index("year")] = "year"
    data.columns = names
    if data.empty:
        raise ValueError("CSV must contain at least one data row.")

    years = []
    for value in data["year"]:
        try:
            number = Decimal(str(value).strip())
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError
            years.append(int(number))
        except (InvalidOperation, ValueError, OverflowError):
            raise ValueError("'year' must contain non-empty integer values.") from None
    if len(set(years)) != len(years):
        raise ValueError("Duplicate years are not allowed.")
    ordered_years = sorted(years)
    if any(right - left != 1 for left, right in zip(ordered_years, ordered_years[1:])):
        raise ValueError("Years must be consecutive after sorting.")
    data["year"] = years

    value_columns = [column for column in names if column != "year"]
    for column in value_columns:
        try:
            cleaned = data[column].map(
                lambda value: str(value).strip().removesuffix("%").strip()
            )
            values = pd.to_numeric(cleaned, errors="raise").astype(float)
        except (ValueError, TypeError, OverflowError):
            raise ValueError(
                f"Column '{column}' must contain non-empty finite numeric values."
            ) from None
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Column '{column}' must contain non-empty finite numeric values."
            )
        data[column] = values

    data = data.sort_values("year").reset_index(drop=True)

    if mode == Mode.average:
        targets = data[value_columns].to_numpy(dtype=float)
        corrected = smooth_average(targets, value_columns)
        rows = [
            [year, month, *corrected[index * 12 + month - 1].tolist()]
            for index, year in enumerate(data["year"].tolist())
            for month in range(1, 13)
        ]
        return _checked_monthly_frame(rows, value_columns)
    rows = []
    previous = None
    for record in data.to_dict(orient="records"):
        current = [record[column] for column in value_columns]
        for month in range(1, 13):
            if mode == Mode.average:
                monthly = current.copy()
            elif previous is None or month == 12:
                # Assign December directly so it exactly matches the input level.
                monthly = current.copy()
            else:
                monthly = []
                for old, new in zip(previous, current):
                    difference = new - old
                    if math.isfinite(difference):
                        value = old + (difference / 12) * month
                    else:
                        # Equivalent interpolation avoids overflow for huge levels.
                        value = old * ((12 - month) / 12) + new * (month / 12)
                    monthly.append(value)
            rows.append([record["year"], month, *monthly])
        previous = current
    result = _checked_monthly_frame(rows, value_columns)
    return result


def _checked_monthly_frame(rows: list, value_columns: list[str]) -> pd.DataFrame:
    """Reject arithmetic overflow before any output formatter sees the data."""
    result = pd.DataFrame(rows, columns=["year", "month", *value_columns])
    for position, column in enumerate(value_columns, start=2):
        if not all(math.isfinite(value) for value in result.iloc[:, position]):
            raise ValueError(
                f"Column '{column}' produced a non-finite result; input values are too large."
            )
    return result


def _read_csv(content: bytes) -> pd.DataFrame:
    """Read UTF-8 CSV, rejecting malformed records and duplicate headers."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeError:
        raise ValueError("CSV could not be decoded; use UTF-8 encoding.") from None
    if not text.strip():
        raise ValueError("CSV must contain at least one data row.")
    if "\x00" in text:
        raise ValueError("CSV contains invalid null characters; use UTF-8 encoding.")
    try:
        records = [row for row in csv.reader(StringIO(text, newline=""), strict=True) if row]
        if not records:
            raise ValueError("CSV must contain at least one data row.")
        width = len(records[0])
        if any(len(row) != width for row in records[1:]):
            raise ValueError("Every CSV row must have the same number of fields as the header.")
        # Read the header as data to prevent pandas from renaming duplicate headers.
        frame = pd.read_csv(
            StringIO(text, newline=""), header=None, dtype=str, keep_default_na=False
        )
        frame.columns = frame.iloc[0].tolist()
        return frame.iloc[1:].reset_index(drop=True)
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"CSV could not be parsed: {exc}") from None


CONVERSION_RESPONSES = {
    200: {
        "content": {"text/csv": {}, "application/json": {}, "application/pdf": {}},
        "description": "Monthly percentages as CSV, JSON, or PDF",
    },
    400: {"description": "Invalid CSV input"},
}


@app.get("/convert", response_class=Response, responses=CONVERSION_RESPONSES)
@app.post("/convert", response_class=Response, responses=CONVERSION_RESPONSES)
def convert(
    file: UploadFile = File(..., description="UTF-8 CSV containing yearly percentages, with or without %"),
    mode: Mode = Query(..., description="No range bounds or penalties; positive first month and exact yearly means (average), or year-end interpolation (exit)"),
    format: OutputFormat = Query(OutputFormat.csv, description="Response format"),
) -> Response:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must have a .csv extension.")
    try:
        try:
            content = file.file.read()
        except OSError:
            raise ValueError("Uploaded CSV could not be read.") from None
        result = yearly_to_monthly(_read_csv(content), mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if format == OutputFormat.json:
        return to_json_response(result)
    if format == OutputFormat.pdf:
        return to_pdf_response(result, mode.value)
    return to_csv_response(result, mode.value)
