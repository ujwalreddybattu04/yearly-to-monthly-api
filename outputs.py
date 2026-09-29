"""Formatting and response builders for monthly percentage data."""
from calendar import month_abbr
from io import BytesIO
from itertools import groupby
from pathlib import Path
from threading import Lock
from xml.sax.saxutils import escape

import pandas as pd
from fastapi.responses import JSONResponse, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import get_data_path
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, TableStyle

_plot_lock = Lock()
FONT_PATH = Path(__file__).resolve().parent / "fonts" / "UnicodeSans-Regular.ttf"
PDF_FONT_NAME = "APIUnicodeSans"
pdfmetrics.registerFont(TTFont(PDF_FONT_NAME, str(FONT_PATH)))
# Matplotlib ships DejaVu Sans; use the same file as a fallback in both renderers
# for extended Latin/Greek/Cyrillic glyphs missing from the CJK font.
FALLBACK_FONT_PATH = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
FALLBACK_PDF_FONT_NAME = "APIFallbackSans"
pdfmetrics.registerFont(TTFont(FALLBACK_PDF_FONT_NAME, str(FALLBACK_FONT_PATH)))
fontManager.addfont(str(FONT_PATH))
CHART_FONT_FAMILIES = [FontProperties(fname=str(FONT_PATH)).get_name(), "DejaVu Sans"]
_PRIMARY_GLYPHS = pdfmetrics.getFont(PDF_FONT_NAME).face.charToGlyph


def _pdf_paragraph(text: str, style) -> Paragraph:
    """Escape literal text and choose a supported font for each character run."""
    def font_for(character):
        return PDF_FONT_NAME if ord(character) in _PRIMARY_GLYPHS else FALLBACK_PDF_FONT_NAME

    markup = "".join(
        f'<font name="{font}">{escape("".join(characters))}</font>'
        for font, characters in groupby(str(text), key=font_for)
    )
    return Paragraph(markup, style)



def format_percentage(value: float) -> str:
    text = f"{value:.2f}"
    if text == "-0.00":
        text = "0.00"
    return text.removesuffix(".00") + "%"


def formatted_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Format value columns by position, preserving even a value named month."""
    rows = [
        [int(row[0]), int(row[1]), *(format_percentage(value) for value in row[2:])]
        for row in df.itertuples(index=False, name=None)
    ]
    return pd.DataFrame(rows, columns=df.columns)


def to_csv_response(df: pd.DataFrame, mode: str) -> Response:
    return Response(
        formatted_monthly(df).to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="monthly_{mode}.csv"'},
    )


def to_json_response(df: pd.DataFrame) -> JSONResponse:
    formatted = formatted_monthly(df)
    # JSON object keys must be unique. Preserve the generated month key and
    # give a colliding input value column a deterministic, non-colliding alias.
    names = list(formatted.columns)
    used = set(names)
    for index in range(2, len(names)):
        if names[index] in names[:index]:
            alias = names[index] + "_value"
            while alias in used:
                alias += "_value"
            names[index] = alias
            used.add(alias)
    records = [dict(zip(names, row)) for row in formatted.itertuples(index=False, name=None)]
    return JSONResponse(records)


def build_trend_chart(df: pd.DataFrame) -> Figure:
    """Build a chart using full-precision values and evenly spaced month indices.

    Call under _plot_lock when used concurrently by request handlers.
    """
    figure = Figure(figsize=(10, 3.8), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = list(range(len(df)))
    for index in range(2, len(df.columns)):
        axis.plot(x, df.iloc[:, index].tolist(), linewidth=1.8, label=str(df.columns[index]))
    # Thin tick labels on long series, without dropping any monthly data points.
    stride = max(1, (len(df) + 11) // 12)
    ticks = sorted(set(range(0, len(df), stride)) | {len(df) - 1})
    labels = [
        f"{month_abbr[int(df.iloc[index, 1])]} {int(df.iloc[index, 0])}"
        for index in ticks
    ]
    axis.set_xticks(ticks, labels, rotation=40, ha="right")
    axis.set_xlabel("Month")
    axis.set_ylabel("Percentage")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.margins(x=0.01)
    # Set properties on the actual artists instead of mutating global rcParams.
    # This includes legend entries, labels, and both axes' tick/offset text.
    for text in figure.findobj(Text):
        text.set_usetex(False)
        text.set_parse_math(False)
        text.set_fontproperties(FontProperties(family=CHART_FONT_FAMILIES, size=text.get_fontsize()))
    return figure


def build_pdf(df: pd.DataFrame, mode: str) -> bytes:
    """Build an in-memory PDF, independently of FastAPI response handling."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output, pagesize=landscape(A4),
        leftMargin=36, rightMargin=36, topMargin=30, bottomMargin=30,
        title=f"Monthly percentages ({mode})",
    )
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = PDF_FONT_NAME
    story = [_pdf_paragraph(f"Monthly percentages ({mode})", styles["Title"])]
    with _plot_lock:
        figure = build_trend_chart(df)
        chart = BytesIO()
        try:
            figure.savefig(chart, format="png", dpi=130)
        finally:
            figure.clear()
    chart.seek(0)
    story.extend([Image(chart, width=document.width, height=document.width * 0.38), Spacer(1, 12)])
    formatted = formatted_monthly(df)
    # Split wide tables into column groups, keeping year/month in each group.
    for start in range(2, len(df.columns), 6):
        if start > 2:
            story.append(PageBreak())
        indices = [0, 1, *range(start, min(start + 6, len(df.columns)))]
        headers = [_pdf_paragraph(str(df.columns[i]), styles["Normal"]) for i in indices]
        rows = [
            [_pdf_paragraph(str(row[i]), styles["Normal"]) for i in indices]
            for row in formatted.itertuples(index=False, name=None)
        ]
        table = LongTable(
            [headers, *rows], repeatRows=1,
            colWidths=[document.width / len(indices)] * len(indices),
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
    document.build(story)
    return output.getvalue()


def to_pdf_response(df: pd.DataFrame, mode: str) -> Response:
    return Response(
        build_pdf(df, mode),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="monthly_{mode}.pdf"'},
    )
