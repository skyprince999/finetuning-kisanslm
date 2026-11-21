#!/usr/bin/env python
"""
Exploratory Data Analysis for KCC (Kisan Call Centre) JSONL data.

Default input: kcc_data_apple.jsonl
Outputs (in --out dir, default eda_apple/):
  - eda_report.md        : text summary of all findings
  - *.png                : charts
  - top_queries.csv      : most frequent normalized query texts
  - duplicates_sample.csv: sample of exact duplicate (query, answer) pairs

Usage:
  python eda_kcc_apple.py
  python eda_kcc_apple.py --input kcc_data_himachal.jsonl --out eda_himachal
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TEXT_COLS = ["QueryText", "KccAns"]
CAT_COLS = ["StateName", "DistrictName", "BlockName", "Season", "Sector",
            "Category", "Crop", "QueryType"]

STOPWORDS = set("""
a an the of in on for to and or is are was were be been with by at from as
about regarding farmer asked query information apple apples plant plants
tree trees orchard orchards my his her their this that what how which when
asking ask want wants know please give tell me it its into
""".split())

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


# ----------------------------------------------------------------- loading
def load_jsonl(path: Path) -> pd.DataFrame:
    rows, bad = [], 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    df = pd.DataFrame(rows)
    df.attrs["bad_lines"] = bad
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Category fields carry stray tabs/whitespace (e.g. "\tPlant Protection\t")
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    if "CreatedOn" in df.columns:
        # mixed precision (with/without fractional secs) -> must use ISO8601
        df["CreatedOn"] = pd.to_datetime(df["CreatedOn"], format="ISO8601", errors="coerce")
        df["date"] = df["CreatedOn"].dt.date
        df["hour"] = df["CreatedOn"].dt.hour
        df["weekday"] = df["CreatedOn"].dt.day_name()
    for c in TEXT_COLS:
        if c in df.columns:
            s = df[c].fillna("")
            df[f"{c}_chars"] = s.str.len()
            df[f"{c}_words"] = s.str.split().str.len()
            df[f"{c}_script"] = s.map(detect_script)
    return df


def detect_script(text: str) -> str:
    if not text:
        return "empty"
    dev, lat = len(DEVANAGARI.findall(text)), len(LATIN.findall(text))
    if dev == 0 and lat == 0:
        return "other/numeric"
    if dev and lat:
        return "mixed" if min(dev, lat) / max(dev, lat) > 0.2 else (
            "hindi(devanagari)" if dev > lat else "english(latin)")
    return "hindi(devanagari)" if dev else "english(latin)"


def norm_query(text) -> str:
    if text is None or pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r"[^\w\sऀ-ॿ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------- helpers
class Report:
    def __init__(self):
        self.lines = []

    def h(self, title, level=2):
        self.lines.append(f"\n{'#' * level} {title}\n")
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")

    def p(self, text=""):
        self.lines.append(str(text))
        print(text)

    def table(self, obj, max_rows=25):
        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
        obj = obj.head(max_rows)
        md = obj.to_markdown() if _has_tabulate() else "```\n" + obj.to_string() + "\n```"
        self.lines.append(md + "\n")
        print(obj.to_string())

    def save(self, path: Path):
        path.write_text("\n".join(self.lines), encoding="utf-8")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def vc(series: pd.Series, n=20) -> pd.DataFrame:
    counts = series.fillna("<NA>").value_counts()
    out = pd.DataFrame({"count": counts, "pct": (counts / counts.sum() * 100).round(2)})
    return out.head(n)


def barh(series: pd.Series, title: str, path: Path, n=20):
    s = series.head(n)[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(s))))
    ax.barh([str(i)[:60] for i in s.index], s.values, color="#3b7dd8")
    ax.set_title(title)
    ax.set_xlabel("count")
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------- sections
def overview(df, r: Report):
    r.h("1. Dataset overview")
    r.p(f"- Rows: **{len(df):,}**")
    r.p(f"- Columns: **{df.shape[1]}** -> {list(df.columns)}")
    r.p(f"- Malformed JSON lines skipped: {df.attrs.get('bad_lines', 0)}")
    r.p(f"- Memory: {df.memory_usage(deep=True).sum() / 1e6:,.1f} MB")

    r.h("2. Missing / empty / 'NA' values", 3)
    rows = []
    for c in df.columns:
        s = df[c]
        null = int(s.isna().sum())
        if pd.api.types.is_string_dtype(s) or s.dtype == object:
            st = s.astype("string")
            empty = int((st.str.len() == 0).sum())
            na_lit = int(st.str.upper().isin(["NA", "N/A", "NULL", "NONE", "0"]).sum())
        else:
            empty = na_lit = 0
        rows.append({"column": c, "null": null, "empty": empty, "literal_NA": na_lit,
                     "unique": int(s.nunique(dropna=True))})
    r.table(pd.DataFrame(rows).set_index("column"), max_rows=50)


def categorical(df, r: Report, out: Path):
    r.h("3. Categorical distributions")
    for c in CAT_COLS:
        if c not in df.columns:
            continue
        r.h(f"{c} (unique={df[c].nunique()})", 3)
        t = vc(df[c], 15)
        r.table(t)
        if df[c].nunique() > 1:
            barh(t["count"], f"Top {c}", out / f"cat_{c}.png")

    if {"StateName", "DistrictName"} <= set(df.columns):
        r.h("Top districts per state", 3)
        g = (df.groupby(["StateName", "DistrictName"]).size()
             .rename("count").reset_index()
             .sort_values(["StateName", "count"], ascending=[True, False])
             .groupby("StateName").head(5).set_index(["StateName", "DistrictName"]))
        r.table(g, max_rows=60)

    if {"StateName", "QueryType"} <= set(df.columns):
        r.h("QueryType mix by top-6 states (% of state's queries)", 3)
        top_s = df["StateName"].value_counts().head(6).index
        sub = df[df["StateName"].isin(top_s)]
        ct = pd.crosstab(sub["QueryType"], sub["StateName"], normalize="columns") * 100
        top = df["QueryType"].value_counts().head(12).index
        r.table(ct.loc[ct.index.intersection(top), top_s].round(1), max_rows=15)

        # numeric / placeholder codes that look like data-entry artefacts
        codes = df["QueryType"].fillna("").str.fullmatch(r"\d+")
        r.p(f"- QueryType values that are bare numeric codes (e.g. '0', '75'): "
            f"{int(codes.sum()):,} rows ({codes.mean() * 100:.2f}%)")


def temporal(df, r: Report, out: Path):
    if "CreatedOn" not in df.columns:
        return
    r.h("4. Temporal analysis")
    ts = df["CreatedOn"].dropna()
    r.p(f"- Unparseable timestamps: {df['CreatedOn'].isna().sum():,}")
    r.p(f"- Date range: {ts.min()}  ->  {ts.max()}")
    r.p(f"- Distinct days with calls: {df['date'].nunique():,}")

    yearly = df.groupby(df["CreatedOn"].dt.year).size()
    r.h("Queries per year", 3)
    r.table(yearly.rename("count"))

    monthly = df.set_index("CreatedOn").resample("MS").size()
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly.plot(ax=ax, color="#3b7dd8")
    ax.set_title("Queries per month")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out / "time_monthly.png", dpi=120)
    plt.close(fig)

    # seasonality: month-of-year, aggregated over all years
    moy = df.groupby(df["CreatedOn"].dt.month).size()
    r.h("Seasonality – queries by calendar month (all years)", 3)
    r.table(moy.rename("count"))

    if "QueryType" in df.columns:
        top = df["QueryType"].value_counts().head(6).index
        piv = (df[df["QueryType"].isin(top)]
               .groupby([df["CreatedOn"].dt.month, "QueryType"]).size().unstack(fill_value=0))
        fig, ax = plt.subplots(figsize=(11, 5))
        piv.plot(ax=ax, marker="o")
        ax.set_title("Seasonality of top query types (by calendar month)")
        ax.set_xlabel("month")
        ax.set_xticks(range(1, 13))
        fig.tight_layout()
        fig.savefig(out / "time_seasonality_querytype.png", dpi=120)
        plt.close(fig)

    r.h("Hour of day / weekday", 3)
    r.table(df["hour"].value_counts().sort_index().rename("count"), max_rows=24)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    r.table(df["weekday"].value_counts().reindex(order).rename("count"))

    # consistency check: year/month columns vs CreatedOn
    if {"year", "month"} <= set(df.columns):
        mism = ((df["year"] != df["CreatedOn"].dt.year) |
                (df["month"] != df["CreatedOn"].dt.month)) & df["CreatedOn"].notna()
        r.p(f"- Rows where year/month fields disagree with CreatedOn: {int(mism.sum()):,}")


def text_stats(df, r: Report, out: Path):
    r.h("5. Text analysis (QueryText / KccAns)")
    for c in TEXT_COLS:
        if c not in df.columns:
            continue
        r.h(f"{c} length statistics", 3)
        desc = df[[f"{c}_chars", f"{c}_words"]].describe(
            percentiles=[.05, .25, .5, .75, .95, .99]).round(1)
        r.table(desc)
        r.h(f"{c} script / language", 3)
        r.table(vc(df[f"{c}_script"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, c in zip(axes, TEXT_COLS):
        if f"{c}_words" in df.columns:
            s = df[f"{c}_words"].clip(upper=df[f"{c}_words"].quantile(0.99))
            ax.hist(s, bins=60, color="#3b7dd8")
            ax.set_title(f"{c} word count (clipped at p99)")
    fig.tight_layout()
    fig.savefig(out / "text_length_hist.png", dpi=120)
    plt.close(fig)

    # quality flags
    r.h("Answer quality flags", 3)
    ans = df["KccAns"].fillna("")
    flags = {
        "answer empty": (ans.str.len() == 0),
        "answer < 5 words": (df["KccAns_words"] < 5),
        "answer > 300 words": (df["KccAns_words"] > 300),
        "answer mentions call/contact/transfer": ans.str.contains(
            r"\b(?:call|contact|transfer|kvk|helpline|toll free)\b", case=False, regex=True),
        "answer contains phone-like number": ans.str.contains(r"\b\d{10}\b", regex=True),
        "answer == query": (ans.str.lower() == df["QueryText"].fillna("").str.lower()),
    }
    r.table(pd.DataFrame({"rows": {k: int(v.sum()) for k, v in flags.items()},
                          "pct": {k: round(v.mean() * 100, 2) for k, v in flags.items()}}))

    r.h("Shortest answers (sample)", 3)
    short = df.loc[df["KccAns_words"] < 5, ["QueryText", "KccAns"]].head(10)
    r.table(short)


def keywords(df, r: Report, out: Path):
    r.h("6. Frequent terms & queries")
    tok = Counter()
    for t in df["QueryText"].dropna():
        for w in norm_query(t).split():
            if len(w) > 2 and w not in STOPWORDS and not w.isdigit():
                tok[w] += 1
    top_terms = pd.Series(dict(tok.most_common(40)), name="count")
    r.h("Top 40 query terms (stopwords removed)", 3)
    r.table(top_terms, max_rows=40)
    barh(top_terms, "Top query terms", out / "query_terms.png", n=30)

    bigr = Counter()
    for t in df["QueryText"].dropna():
        ws = [w for w in norm_query(t).split() if w not in STOPWORDS and len(w) > 2]
        bigr.update(zip(ws, ws[1:]))
    r.h("Top 25 query bigrams", 3)
    r.table(pd.Series({" ".join(k): v for k, v in bigr.most_common(25)}, name="count"))

    nq = df["QueryText"].map(norm_query)
    top_q = nq.value_counts().head(200).rename("count")
    top_q.to_csv(out / "top_queries.csv", encoding="utf-8")
    r.h("Top 20 normalized queries", 3)
    r.table(top_q, max_rows=20)


def duplicates(df, r: Report, out: Path):
    r.h("7. Duplicates & answer diversity")
    full = df.duplicated(subset=[c for c in df.columns if not c.endswith(("_chars", "_words", "_script"))
                                 and c not in ("date", "hour", "weekday")]).sum()
    qa = df.duplicated(subset=TEXT_COLS).sum()
    nq = df["QueryText"].map(norm_query)
    q_only = nq.duplicated().sum()
    r.p(f"- Fully identical records: {full:,} ({full / len(df) * 100:.2f}%)")
    r.p(f"- Duplicate (QueryText, KccAns) pairs: {qa:,} ({qa / len(df) * 100:.2f}%)")
    r.p(f"- Duplicate normalized QueryText: {q_only:,} ({q_only / len(df) * 100:.2f}%)")
    r.p(f"- Unique normalized queries: {nq.nunique():,}")
    r.p(f"- Unique answers: {df['KccAns'].nunique():,}")

    dups = (df[df.duplicated(subset=TEXT_COLS, keep=False)]
            .groupby(TEXT_COLS).size().rename("count").sort_values(ascending=False))
    dups.head(500).to_csv(out / "duplicates_sample.csv", encoding="utf-8")

    # how many different answers does the same query get?
    tmp = pd.DataFrame({"q": nq, "a": df["KccAns"]})
    per_q = tmp.groupby("q")["a"].nunique()
    freq_q = per_q[tmp["q"].value_counts().reindex(per_q.index) >= 20]
    r.h("Answer diversity for frequent queries (>=20 occurrences)", 3)
    r.p(f"- Frequent queries: {len(freq_q):,}; median distinct answers each: "
        f"{freq_q.median() if len(freq_q) else 0}")
    r.table(freq_q.sort_values(ascending=False).rename("distinct_answers"), max_rows=15)

    r.h("Most reused answers", 3)
    r.table(df["KccAns"].str.slice(0, 120).value_counts().head(15).rename("count"))


def heatmap(df, r: Report, out: Path):
    if not {"DistrictName", "QueryType"} <= set(df.columns):
        return
    top_d = df["DistrictName"].value_counts().head(15).index
    top_q = df["QueryType"].value_counts().head(10).index
    ct = pd.crosstab(df["DistrictName"], df["QueryType"]).loc[top_d, top_q]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(ct_pct.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(top_q)), [q[:25] for q in top_q], rotation=45, ha="right")
    ax.set_yticks(range(len(top_d)), top_d)
    for i in range(ct_pct.shape[0]):
        for j in range(ct_pct.shape[1]):
            ax.text(j, i, f"{ct_pct.iat[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="% of district queries")
    ax.set_title("QueryType share by top-15 districts")
    fig.tight_layout()
    fig.savefig(out / "heatmap_district_querytype.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="EDA for KCC JSONL files")
    ap.add_argument("--input", default="kcc_data_apple.jsonl")
    ap.add_argument("--out", default="eda_apple")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Input not found: {src}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src} ...")
    df = clean(load_jsonl(src))

    r = Report()
    r.lines.append(f"# EDA report: `{src.name}`\n")
    overview(df, r)
    categorical(df, r, out)
    temporal(df, r, out)
    text_stats(df, r, out)
    keywords(df, r, out)
    duplicates(df, r, out)
    heatmap(df, r, out)

    r.save(out / "eda_report.md")
    print(f"\nDone. Report + charts written to: {out.resolve()}")


if __name__ == "__main__":
    main()
