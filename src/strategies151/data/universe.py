"""Index membership lists.

Only used to *choose* which tickers to load; bars themselves always come from
the ``stooq.daily`` table.
"""

from __future__ import annotations

import csv
import io
import logging
import re

import requests

log = logging.getLogger(__name__)

_UA = "Mozilla/5.0"
GITHUB_CONSTITUENTS = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/"
    "data/constituents.csv"
)
WIKIPEDIA_LIST = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class UniverseError(RuntimeError):
    pass


def _from_github(timeout: float) -> list[dict]:
    resp = requests.get(GITHUB_CONSTITUENTS, headers={"User-Agent": _UA}, timeout=timeout)
    resp.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(resp.text)))
    return [
        {"ticker": r["Symbol"].strip(), "name": r.get("Security", "").strip(),
         "sector": r.get("GICS Sector", "").strip()}
        for r in rows
        if r.get("Symbol")
    ]


def _from_wikipedia(timeout: float) -> list[dict]:
    resp = requests.get(WIKIPEDIA_LIST, headers={"User-Agent": _UA}, timeout=timeout)
    resp.raise_for_status()
    table = re.search(r'<table[^>]*id="constituents".*?</table>', resp.text, re.S)
    if not table:
        raise UniverseError("could not locate the constituents table")
    rows = re.findall(r"<tr>(.*?)</tr>", table.group(0), re.S)
    out = []
    for row in rows[1:]:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        if len(cells) < 4:
            continue
        clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        out.append({"ticker": clean[0], "name": clean[1], "sector": clean[2]})
    return out


def sp500_constituents(timeout: float = 30.0) -> list[dict]:
    """Current S&P 500 membership: ticker, company name and GICS sector.

    This is *today's* membership. Applying it to a decade of history is
    survivorship-biased - companies that were dropped from the index, or that
    failed, are absent - which flatters any long-only result. The bias is
    unavoidable without a point-in-time membership file and is recorded in the
    run context so it travels with the numbers.
    """
    errors = []
    for source, fetch in (("github", _from_github), ("wikipedia", _from_wikipedia)):
        try:
            rows = fetch(timeout)
            if len(rows) > 400:
                log.info("loaded %d S&P 500 constituents from %s", len(rows), source)
                return rows
            errors.append(f"{source}: only {len(rows)} rows")
        except Exception as exc:  # noqa: BLE001 - fall through to the next source
            errors.append(f"{source}: {exc}")
    raise UniverseError(f"could not fetch the constituent list -> {errors}")


def to_yahoo_symbol(ticker: str) -> str:
    """``BRK.B`` -> ``BRK-B``; Yahoo uses a hyphen for share classes."""
    return ticker.strip().upper().replace(".", "-")


def to_db_symbol(ticker: str) -> str:
    """``brk.b`` -> ``BRK-B``: the spelling the bar table stores.

    The table is filled from stooq, which hyphenates share classes, while index
    membership lists write ``BRK.B``. Normalising at the database boundary keeps
    one spelling of a name across loads, reads and the panel.
    """
    return ticker.strip().upper().replace(".", "-")


def to_stooq_symbol(ticker: str) -> str:
    """``BRK.B`` -> ``brk-b``; stooq also hyphenates share classes."""
    return ticker.strip().lower().replace(".", "-")
