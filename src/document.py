"""Read stock tickers out of an uploaded document (PDF).

Tickers are found three ways, in order of appearance:
1. A table column headed "Ticker", "Symbol", "Scrip" or "Stock code".
2. Anywhere in the text, symbols with a Yahoo Finance exchange suffix
   (RELIANCE.NS, 500325.BO, VOD.L, 0700.HK, ...), labelled symbols
   ("Ticker: AAPL") and cashtags ($MSFT).
3. Nothing else: plain capitalised words in prose are ignored, so "NSE" or
   "IT Services" are never mistaken for tickers.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

MAX_PAGES = 50
YAHOO_SUFFIXES = (
    "NS", "BO", "L", "TO", "V", "AX", "HK", "DE", "F", "PA", "AS", "MI", "SW", "T", "SI", "KS", "KQ",
    "SS", "SZ", "TW", "JK", "SA", "MX", "ST", "OL", "CO", "HE", "BR", "LS", "MC", "NZ", "IR", "VI",
    "TA", "JO", "BK", "KL",
)
STOPWORDS = {
    "NSE", "BSE", "NYSE", "NASDAQ", "LSE", "TICKER", "TICKERS", "SYMBOL", "SYMBOLS", "TOTAL", "NA", "N/A",
    "INR", "USD", "EUR", "GBP", "JPY", "ETF", "IPO", "CEO", "CFO", "PDF", "AI", "IT", "LTD", "INC", "PLC",
    "QTY", "BUY", "SELL", "HOLD", "DATE", "NOTE", "NOTES", "SCRIP", "CODE",
}

_SYM = r"[A-Z0-9][A-Z0-9&\-]{0,14}"
_SUFFIXED_RE = re.compile(rf"(?<![A-Za-z0-9.&\-])({_SYM}\.(?:{'|'.join(YAHOO_SUFFIXES)}))(?![A-Za-z0-9&\-])")
_LABELLED_RE = re.compile(
    rf"(?i:\b(?:ticker|symbol|nse symbol|bse symbol|stock code|scrip(?: code)?)\s*[:=]\s*)"
    rf"(\^?{_SYM}(?:\.[A-Z]{{1,3}})?)(?![A-Za-z0-9&\-])"
)
_CASHTAG_RE = re.compile(r"(?<![A-Za-z0-9])\$([A-Z]{1,6}(?:\.[A-Z]{1,3})?)(?![A-Za-z0-9])")
_HEADER_RE = re.compile(r"\b(?:ticker|symbol|scrip|stock code|nse code|bse code)s?\b", re.I)
_TICKER_TOKEN_RE = re.compile(rf"^\^?{_SYM}(?:\.[A-Z]{{1,3}})?$")
_TABLE_GAP_RE = re.compile(r"\S {2,}\S")  # table rows have multi-space column gaps


class DocumentError(ValueError):
    """Raised when a document cannot be read or contains no usable text."""


@dataclass
class DocumentTickers:
    tickers: list[str]
    text: str
    pages: int


def _is_ticker(token: str) -> bool:
    return (
        bool(_TICKER_TOKEN_RE.match(token))
        and any(ch.isalpha() for ch in token)
        and token.upper() not in STOPWORDS
    )


def _column_tickers(lines: list[str], line_offsets: list[int]) -> list[tuple[int, str]]:
    """Tickers listed under a Ticker/Symbol column header in layout-preserved text."""
    found: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        header = _HEADER_RE.search(line)
        if not header or not _TABLE_GAP_RE.search(line):
            continue
        col_start, col_end = header.start(), max(header.end(), header.start() + 4)
        pending: tuple[int, str] | None = None
        for j in range(i + 1, len(lines)):
            cell = next(
                (m for m in re.finditer(r"\S+", lines[j]) if m.start() < col_end and m.end() > col_start), None
            )
            if cell is None:
                continue  # empty cell: wrapped text from another column
            token = cell.group().strip(",;:")
            # Rejoin a ticker the PDF wrapped inside its cell, e.g. "HDFCBANK.N" + "S".
            if pending and re.fullmatch(r"[A-Z]{1,2}", token):
                found.append((pending[0], pending[1] + token))
                pending = None
                continue
            if pending:
                found.append(pending)
                pending = None
            if _HEADER_RE.fullmatch(token):
                break  # next table's header; it is handled by the outer loop
            if re.fullmatch(r"[A-Z0-9&\-]+\.[A-Z]?", token):
                pending = (line_offsets[j] + cell.start(), token)
            elif _is_ticker(token):
                found.append((line_offsets[j] + cell.start(), token))
            elif any(ch.islower() for ch in token):
                break  # reached prose: the table has ended
        if pending:
            found.append(pending)
    return found


def extract_tickers_from_text(text: str) -> list[str]:
    """Return tickers found in `text`, de-duplicated, in order of appearance."""
    lines = text.splitlines()
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    hits = _column_tickers(lines, offsets)
    for pattern in (_SUFFIXED_RE, _LABELLED_RE, _CASHTAG_RE):
        hits += [(m.start(1), m.group(1)) for m in pattern.finditer(text)]

    seen: set[str] = set()
    ordered: list[str] = []
    for _, ticker in sorted(hits, key=lambda h: h[0]):
        ticker = ticker.upper()
        if ticker not in seen and _is_ticker(ticker):
            seen.add(ticker)
            ordered.append(ticker)
    return ordered


def read_pdf_text(data: bytes) -> tuple[list[str], int]:
    """Layout-preserving text of each page (falls back to plain extraction)."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = list(reader.pages)[:MAX_PAGES]
    except Exception as exc:
        raise DocumentError(f"Could not open the PDF: {exc}") from exc

    texts = []
    for page in pages:
        try:
            texts.append(page.extract_text(extraction_mode="layout") or "")
        except Exception:
            texts.append(page.extract_text() or "")
    return texts, len(reader.pages)


def extract_tickers_from_pdf(data: bytes) -> DocumentTickers:
    page_texts, n_pages = read_pdf_text(data)
    if not any(t.strip() for t in page_texts):
        raise DocumentError(
            "No text found in this PDF. It may be a scanned image; please upload a PDF with selectable text."
        )
    tickers: list[str] = []
    for page_text in page_texts:  # per page, so a table header applies to its own page
        tickers += [t for t in extract_tickers_from_text(page_text) if t not in tickers]
    return DocumentTickers(tickers=tickers, text="\n\n".join(page_texts), pages=n_pages)


def apply_default_suffix(tickers: list[str], suffix: str) -> list[str]:
    """Add an exchange suffix (e.g. ".NS") to symbols that have none."""
    if not suffix:
        return list(tickers)
    out: list[str] = []
    for t in tickers:
        t2 = t if ("." in t or t.startswith("^")) else t + suffix
        if t2 not in out:
            out.append(t2)
    return out


if __name__ == "__main__":
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "examples" / "sample_watchlist.pdf"
    doc = extract_tickers_from_pdf(example.read_bytes())
    print(f"example PDF ({doc.pages} page): {doc.tickers}")
    assert doc.tickers == ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "MSFT"], doc.tickers

    cases = {
        "prose with suffixes, labels and cashtags": (
            "We like M&M.NS and BAJAJ-AUTO.NS on the NSE, Ticker: AAPL, plus $MSFT. "
            "Use .NS for NSE and .BO for BSE. IT Services and AI are hot.",
            ["M&M.NS", "BAJAJ-AUTO.NS", "AAPL", "MSFT"],
        ),
        "broker table without suffixes, wrapped cell": (
            "Symbol        Qty    Avg price\n\nRELIANCE      10     1250.00\nHDFCBANK.N    5      1600.00\nS\n"
            "TOTAL                  2850.00\n\nNotes: prices in INR.",
            ["RELIANCE", "HDFCBANK.NS"],
        ),
        "prose mentioning the word ticker": (
            "Upload this and the app reads the Ticker column of the table.\nIt then downloads data.",
            [],
        ),
    }
    for name, (text, expected) in cases.items():
        got = extract_tickers_from_text(text)
        status = "OK " if got == expected else "FAIL"
        print(f"{status} {name}: {got}")
        assert got == expected, (name, got)

    assert apply_default_suffix(["RELIANCE", "AAPL.NS", "^NSEI"], ".NS") == ["RELIANCE.NS", "AAPL.NS", "^NSEI"]
    try:
        extract_tickers_from_pdf(b"not a pdf")
    except DocumentError as e:
        print("OK  bad file ->", str(e)[:60])
    print("document checks passed")
