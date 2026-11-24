"""Generate examples/sample_watchlist.pdf, the example document for the upload page.

Run from the project root:  python examples/make_sample_pdf.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

OUT = Path(__file__).with_name("sample_watchlist.pdf")

ROWS = [
    ("Ticker", "Company", "Exchange", "Sector", "Why we are watching"),
    ("RELIANCE.NS", "Reliance Industries", "NSE", "Energy & Telecom", "Largest Indian company by market cap"),
    ("TCS.NS", "Tata Consultancy Services", "NSE", "IT Services", "Bellwether for Indian IT spending"),
    ("INFY.NS", "Infosys", "NSE", "IT Services", "Compare with TCS"),
    ("HDFCBANK.NS", "HDFC Bank", "NSE", "Banking", "Largest private-sector bank"),
    ("AAPL", "Apple Inc.", "NASDAQ", "Technology", "Global large-cap reference"),
    ("MSFT", "Microsoft Corp.", "NASDAQ", "Technology", "Cloud and AI exposure"),
]


def build(path: Path = OUT) -> Path:
    pdf = FPDF(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    pdf.set_title("Stock Watchlist")

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(text="Stock Watchlist", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(
        text=f"Prepared {date.today():%d %B %Y}  |  For the TimesFM Stock Forecast app",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(6)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0, 6,
        "This document lists the stocks we want to forecast. Upload it on the Upload document page: "
        "the app reads the Ticker column, downloads the latest daily prices from Yahoo Finance and "
        "predicts the coming trading days for each stock.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    with pdf.table(col_widths=(31, 44, 20, 28, 51), line_height=7, text_align="LEFT") as table:
        for row in ROWS:
            r = table.row()
            for cell in row:
                r.cell(cell)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(text="Notes", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 5.5,
        "Indian stocks use the .NS suffix for the NSE or .BO for the BSE. US stocks use the plain "
        "symbol. Review the list on the upload page before predicting; you can remove or add tickers there.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(
        0, 5,
        "Educational tool only. Predictions are not financial advice. Stock prices are affected by news "
        "and events the model cannot see.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    pdf.output(str(path))
    return path


if __name__ == "__main__":
    print(f"wrote {build()}")
