"""Extract concession terms from advertised specials text, and score the result.

Phase 2 step 2.1 (docs/project_phases.md). Reads the source tables rather than the
master table: the master keeps only the first special per listing, while
market.concession_event holds all 91,156 and truth.concession carries the labels
plus text_style and text_source, which the results are broken down by.

What is and is not a real task here:
  --Type is already solved. market.concession_event.type agrees with
    truth.concession.type on 100% of rows, so a type classifier scores ~1.0 by
    reading the column next to the text. It is reported, not modelled.
  --Value is the real task. The amount, its basis (months free vs dollars) and the
    lease term have to come out of the prose.

Texts are LLM-written, so rules can fit that generator's phrasing rather than
English. Every metric is therefore reported by text_style and text_source.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "accolade.duckdb"

WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "a": 1, "an": 1, "half": 0.5, "1st": 1, "first": 1, "second": 2}
# The generator converts at exactly four weeks to the month: its "4 weeks free"
# is labelled 1.0 months and "6 weeks free" 1.5. A calendar 4.345 misses both.
WEEKS_PER_MONTH = 4.0

# The generator writes U+2011 non-breaking hyphens ("12-month"), en and em dashes,
# and curly apostrophes. Without folding them a plain [-] class misses the term.
DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
QUOTES = {ord("’"): "'", ord("‘"): "'", ord("“"): '"', ord("”"): '"'}


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).translate(DASHES).translate(QUOTES)
    return re.sub(r"\s+", " ", t).strip().lower()


NUM = r"(\d+(?:[.,]\d+)?)"
HALF = r"(?:half|1/2|one[- ]half)"

MONTHS_PATTERNS = [
    rf"{HALF}\s+(?:a\s+)?month(?:'s)?\s+(?:of\s+)?(?:rent\s+)?free",
    rf"(?:up to\s+)?{NUM}\s*\+?\s*months?\s+(?:of\s+)?(?:rent\s+)?free",
    rf"(?:up to\s+)?{NUM}\s*weeks?\s+(?:of\s+)?(?:rent\s+)?free",
    rf"free\s+rent\s+for\s+(?:up to\s+)?{NUM}\s*months?",
    r"(first|one|two|three)\s+months?\s+(?:of\s+)?(?:rent[- ])?free",
    r"\b(a|an)\s+month(?:'s)?\s+(?:of\s+)?(?:rent\s+)?free",
    r"get\s+(a|an)\s+month\s+free",
    r"free\s+rent\s+(?:on|for)\s+(?:the\s+)?(first|one|two|three)\s+months?",
    r"(first)\s+month\s+(?:is\s+)?(?:on us|rent[- ]free|free)",
    r"your\s+(first)\s+month\s+(?:on us|free|rent[- ]free)",
    r"free\s+rent\s+on\s+(first)\s+month",
    rf"(?:up to\s+)?{NUM}\s*months?\s+free",
]
DOLLAR_PATTERNS = [
    r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:off|credit|discount|rebate|back)",
    r"(?:credit|discount|rebate|off)\s+of\s+\$\s?([\d,]+(?:\.\d+)?)",
    r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:move[- ]in\s+)?(?:credit|bonus)",
    r"\$\s?([\d,]+(?:\.\d+)?)\s*per\s+month",
    r"\$\s?([\d,]+(?:\.\d+)?)",
]
TERM_PATTERNS = [
    rf"{NUM}\s*-?\s*month\s+lease",
    rf"{NUM}\s*-?\s*month\s+leases",
    rf"on\s+a\s+{NUM}\s*-?\s*month",
    rf"lease\s+term\s+of\s+{NUM}\s*months?",
]


def _num(tok: str) -> float | None:
    if tok is None:
        return None
    t = tok.strip().lower().replace(",", "")
    if t in WORD_NUM:
        return float(WORD_NUM[t])
    try:
        return float(t)
    except ValueError:
        return None


def extract(text: str) -> dict:
    """Pull amount, basis and lease term out of one specials text."""
    t = norm(text)
    out = {"pred_amount": None, "pred_basis": None, "pred_term": None}

    for i, pat in enumerate(MONTHS_PATTERNS):
        m = re.search(pat, t)
        if not m:
            continue
        if i == 0:
            val = 0.5
        else:
            val = _num(m.group(1)) if m.groups() else None
            if val is None:
                continue
            if "week" in pat:
                val = round(val / WEEKS_PER_MONTH, 3)
        out.update(pred_amount=val, pred_basis="months_free")
        break

    if out["pred_basis"] is None:
        for pat in DOLLAR_PATTERNS:
            m = re.search(pat, t)
            if m:
                val = _num(m.group(1))
                if val is not None:
                    out.update(pred_amount=val, pred_basis="dollars")
                break

    for pat in TERM_PATTERNS:
        m = re.search(pat, t)
        if m:
            v = _num(m.group(1))
            if v and 1 <= v <= 36:
                out["pred_term"] = v
                break
    return out


def predict_total(row) -> float | None:
    """Money value of the concession, given the posted rent and the term."""
    amt, basis = row.pred_amount, row.pred_basis
    if amt is None or pd.isna(amt):
        return None
    term = row.pred_term if pd.notna(row.pred_term) else 12.0
    if basis == "months_free":
        return amt * row.asking_rent_at_event if pd.notna(row.asking_rent_at_event) else None
    if row.type == "recurring_discount":
        return amt * term
    return amt


def load(limit: int | None) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"attach '{DB.as_posix()}' as acc (read_only)")
    sql = """
        SELECT ce.concession_event_id, ce.specials, ce.type AS observed_type,
               tc.type, tc.value_amount, tc.value_basis, tc.lease_term_months,
               tc.lease_term_stated, tc.total_value, tc.asking_rent_at_event,
               tc.is_generic, tc.text_style, tc.text_source
        FROM acc.market.concession_event ce
        JOIN acc.truth.concession tc USING (concession_event_id)
    """
    if limit:
        sql += f" USING SAMPLE reservoir({limit} ROWS) REPEATABLE (42)"
    return con.execute(sql).df()


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row correctness flags, so any breakdown is just a groupby."""
    d = df.copy()
    d["basis_ok"] = d.pred_basis == d.value_basis
    amt_close = np.isclose(d.pred_amount.astype(float), d.value_amount.astype(float),
                           rtol=0.01, equal_nan=False)
    d["amount_ok"] = d.basis_ok & amt_close
    d["found_any"] = d.pred_amount.notna()

    d["term_ok"] = np.where(
        d.lease_term_stated,
        d.pred_term.astype(float) == d.lease_term_months.astype(float),
        d.pred_term.isna())

    d["pred_total"] = d.apply(predict_total, axis=1)
    err = (d.pred_total - d.total_value).abs()
    d["total_abs_err"] = err
    d["total_pct_err"] = np.where(d.total_value.abs() > 0, err / d.total_value.abs(), np.nan)
    d["total_ok"] = d.total_pct_err <= 0.05
    return d


def summarize(d: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    g = d.groupby(by, dropna=False) if by else [("all", d)]
    rows = []
    for key, part in (g if by else g):
        rows.append({
            (by or "slice"): key,
            "n": len(part),
            "found_pct": round(100 * part.found_any.mean(), 1),
            "basis_acc": round(100 * part.basis_ok.mean(), 1),
            "amount_acc": round(100 * part.amount_ok.mean(), 1),
            "term_acc": round(100 * part.term_ok.mean(), 1),
            "total_within_5pct": round(100 * part.total_ok.mean(), 1),
            "total_median_pct_err": round(100 * part.total_pct_err.median(), 1),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="sample N events; default all")
    ap.add_argument("--out-dir", default=str(REPO / "outputs"))
    ap.add_argument("--prefix", default="01_phase_02")
    ap.add_argument("--show-misses", type=int, default=8)
    a = ap.parse_args()

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 70)
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load(a.limit)
    print(f"{len(df):,} concession events\n")

    agree = (df.observed_type == df.type).mean()
    print(f"Type is not a modelling task: the observed type column already agrees with "
          f"the answer key on {100 * agree:.1f}% of rows.\n")

    scored = extract_frame(df)
    real = scored[~scored.is_generic]
    generic = scored[scored.is_generic]
    print(f"{len(generic):,} generic teasers carry no terms and are excluded "
          f"({100 * len(generic) / len(scored):.1f}% of events)\n")

    ev = evaluate(real)
    print("Overall, non-generic events")
    print(summarize(ev).to_string(index=False))
    for by in ("type", "text_style", "text_source"):
        print(f"\nBy {by}")
        print(summarize(ev, by).sort_values("n", ascending=False).to_string(index=False))

    ev.drop(columns=["specials"]).to_csv(outdir / f"{a.prefix}_concession_extraction_rows.csv", index=False)
    summary = pd.concat([summarize(ev).assign(slice_by="overall")] +
                        [summarize(ev, b).rename(columns={b: "slice"}).assign(slice_by=b)
                         for b in ("type", "text_style", "text_source")], ignore_index=True)
    summary.to_csv(outdir / f"{a.prefix}_concession_extraction_summary.csv", index=False)

    if a.show_misses:
        print(f"\nWhere the rules fail (first {a.show_misses})")
        miss = ev[~ev.amount_ok].head(a.show_misses)
        for _, r in miss.iterrows():
            print(f"  [{r.type} truth {r.value_amount} {r.value_basis} | "
                  f"pred {r.pred_amount} {r.pred_basis}] {norm(r.specials)[:110]}")
    print(f"\nwrote 2 CSVs to {outdir}")


def extract_frame(df: pd.DataFrame) -> pd.DataFrame:
    ext = pd.DataFrame([extract(t) for t in df.specials], index=df.index)
    return pd.concat([df, ext], axis=1)


if __name__ == "__main__":
    main()
