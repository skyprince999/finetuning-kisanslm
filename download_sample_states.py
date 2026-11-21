#!/usr/bin/env python3
"""
download_sample_states.py

Downloads a proportionally stratified sample of the KCC dataset.

Strategy:
  1. Probe the API for each state's total record count.
  2. Compute each state's share of the grand total.
  3. Allocate TARGET_SAMPLE_SIZE records across states in the same ratio.
  4. For each state, pick a random starting offset within its range so the
     sample is spread across the dataset rather than always taken from the top.
  5. Fetch consecutive pages from that offset until the state's quota is met.
  6. Write per-state files: kcc_sample_{state_name}.jsonl
  7. Write a combined file: kcc_sample_combined.jsonl
  8. Write a sampling report: kcc_sample_report.md / .csv / .json

The result mirrors the original state-level distribution at a fraction of the size.
"""

import os
import csv
import json
import math
import time
import random
import requests
from datetime import date
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from typing import Optional

# ---------- CONFIG ----------
RESOURCE_ID = "cef25fe2-9231-4128-8aec-2c948fedd43f"
API_ENDPOINT = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
API_KEY = os.getenv("KCC_API_KEY", "579b464db66ec23bdd000001e512631e0be6419550aad2c1f612a8e5")

# Total records you want in the final sample across all states combined.
# States with 0 records are skipped automatically.
TARGET_SAMPLE_SIZE = 100_000

# When True, the starting offset for each state is chosen randomly within its
# available range, so the sample is not always from the beginning of the data.
# Set to False to always start from offset 0 (deterministic, easier to reproduce).
RANDOM_OFFSET = True

PAGE_LIMIT = 200
REQUEST_TIMEOUT = (5, 90)

RETRY_TOTAL = 8
RETRY_BACKOFF_FACTOR = 1.0
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)

MAX_ATTEMPTS_PER_PAGE = 6
BASE_PAUSE = 0.4

OUTPUT_DIR = "."
COMBINED_FILE = os.path.join(OUTPUT_DIR, "kcc_sample_combined.jsonl")
REPORT_CSV = "kcc_sample_report.csv"
REPORT_JSON = "kcc_sample_report.json"
REPORT_MD = "kcc_sample_report.md"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "kcc-dataset-sampler/1.0",
    "Connection": "keep-alive",
}

STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
]
# ----------------------------


# ------------------------------------------------------------------ #
#  Helpers shared with download_all_states.py                         #
# ------------------------------------------------------------------ #

def state_to_sample_filename(state: str) -> str:
    slug = state.lower().replace(" ", "_")
    return os.path.join(OUTPUT_DIR, f"kcc_sample_{slug}.jsonl")


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def build_params(state: str, offset: int, limit: int) -> dict:
    return {
        "api-key": API_KEY,
        "format": "json",
        "offset": offset,
        "limit": limit,
        "filters[StateName]": state.upper(),
    }


def append_jsonl(path: str, records: list):
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_records(data) -> list:
    if isinstance(data, dict):
        for key in ("records", "data", "items", "results", "response"):
            if key in data and isinstance(data[key], list):
                return data[key]
        if "result" in data and isinstance(data["result"], dict):
            for key in ("records", "data", "items", "results"):
                if key in data["result"] and isinstance(data["result"][key], list):
                    return data["result"][key]
        return []
    if isinstance(data, list):
        return data
    return []


def extract_total(data) -> Optional[int]:
    if not isinstance(data, dict):
        return None
    if "total" in data and isinstance(data["total"], int):
        return data["total"]
    if "count" in data and isinstance(data["count"], int):
        return data["count"]
    if "result" in data and isinstance(data["result"], dict):
        val = data["result"].get("total") or data["result"].get("count")
        if val is not None:
            return int(val)
    return None


# ------------------------------------------------------------------ #
#  Phase 1 — collect totals for all states                            #
# ------------------------------------------------------------------ #

def fetch_total(session: requests.Session, state: str) -> tuple[Optional[int], Optional[list]]:
    """Return (total_records, field_names) for the state via a limit=1 probe."""
    params = build_params(state, offset=0, limit=1)
    try:
        resp = session.get(API_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            print(f"  [{state}] HTTP {resp.status_code} during probe.")
            return None, None
        data = resp.json()
        total = extract_total(data)
        records = extract_records(data)
        fields = list(records[0].keys()) if records else None
        return total, fields
    except Exception as e:
        print(f"  [{state}] Probe error: {e}")
        return None, None


# ------------------------------------------------------------------ #
#  Phase 2 — compute proportional allocation                          #
# ------------------------------------------------------------------ #

def allocate_samples(totals: dict, target: int) -> dict:
    """
    Given {state: total_records}, return {state: sample_size} so that:
      - States with 0 records get 0.
      - The sum of sample sizes equals target (after integer rounding).
      - Each non-zero state gets at least 1 record.
    Uses the largest-remainder method for fair rounding.
    """
    grand_total = sum(v for v in totals.values() if v)
    if grand_total == 0:
        return {s: 0 for s in totals}

    # Exact (floating-point) quotas
    quotas = {
        s: (t / grand_total * target) if t else 0.0
        for s, t in totals.items()
    }

    # Floor allocations
    floors = {s: int(math.floor(q)) for s, q in quotas.items()}
    remainder = target - sum(floors.values())

    # Distribute remaining seats by largest fractional part
    fractions = sorted(
        [(q - math.floor(q), s) for s, q in quotas.items() if totals.get(s, 0) > 0],
        reverse=True,
    )
    for _, s in fractions[:remainder]:
        floors[s] += 1

    # Guarantee at least 1 for every non-zero state
    for s, t in totals.items():
        if t and t > 0 and floors[s] == 0:
            floors[s] = 1

    return floors


# ------------------------------------------------------------------ #
#  Phase 3 — download sample for one state                            #
# ------------------------------------------------------------------ #

def download_sample(
    session: requests.Session,
    state: str,
    state_total: int,
    sample_size: int,
    start_offset: int,
) -> int:
    """
    Fetch exactly `sample_size` records for `state` starting at `start_offset`.
    Writes to the per-state sample file AND appends to COMBINED_FILE.
    Returns the number of records actually written.
    """
    out_path = state_to_sample_filename(state)
    written = 0
    offset = start_offset

    while written < sample_size:
        need = min(PAGE_LIMIT, sample_size - written)
        params = build_params(state, offset=offset, limit=need)
        attempt = 0

        while attempt < MAX_ATTEMPTS_PER_PAGE:
            attempt += 1
            try:
                resp = session.get(API_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError as e:
                        wait = (2 ** attempt) + random.random()
                        print(f"\n  JSON error (attempt {attempt}), wait {wait:.1f}s: {e}")
                        time.sleep(wait)
                        continue

                    records = extract_records(data)
                    if not records:
                        print(f"\n  No records at offset {offset}. Stopping early.")
                        return written

                    # Trim to exactly what we need
                    records = records[:sample_size - written]

                    append_jsonl(out_path, records)
                    append_jsonl(COMBINED_FILE, records)

                    written += len(records)
                    offset += len(records)
                    print(f"  [{state}] {written}/{sample_size}", end="\r")
                    time.sleep(random.uniform(BASE_PAUSE, BASE_PAUSE * 2.0))
                    break

                elif resp.status_code in (429, 503, 504):
                    wait = (2 ** attempt) + random.random()
                    print(f"\n  HTTP {resp.status_code} attempt {attempt}, wait {wait:.1f}s.")
                    time.sleep(wait)

                elif 400 <= resp.status_code < 500:
                    print(f"\n  Client error {resp.status_code}: {resp.text[:300]}")
                    return written

                else:
                    wait = (2 ** attempt) + random.random()
                    print(f"\n  Server error {resp.status_code} attempt {attempt}, wait {wait:.1f}s.")
                    time.sleep(wait)

            except requests.exceptions.Timeout:
                wait = (2 ** attempt) + random.random()
                print(f"\n  Timeout attempt {attempt}, wait {wait:.1f}s.")
                time.sleep(wait)
            except requests.exceptions.RequestException as e:
                wait = (2 ** attempt) + random.random()
                print(f"\n  RequestException ({e}) attempt {attempt}, wait {wait:.1f}s.")
                time.sleep(wait)
        else:
            print(f"\n  Exceeded max attempts at offset {offset}.")
            return written

    print()
    return written


# ------------------------------------------------------------------ #
#  Report writers                                                      #
# ------------------------------------------------------------------ #

def save_reports(rows: list, grand_total: int, target: int):
    # CSV
    fieldnames = ["state", "api_total", "pct_of_total", "sample_target",
                  "sample_downloaded", "start_offset", "fields"]
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "state": r["state"],
                "api_total": r["api_total"] or "",
                "pct_of_total": f"{r['pct_of_total']:.4f}",
                "sample_target": r["sample_target"],
                "sample_downloaded": r["sample_downloaded"],
                "start_offset": r["start_offset"],
                "fields": "|".join(r["fields"]) if r["fields"] else "",
            })

    # JSON
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({"target": target, "grand_total": grand_total, "states": rows},
                  f, ensure_ascii=False, indent=2)

    # Markdown
    actual_total = sum(r["sample_downloaded"] for r in rows)
    lines = [
        "# KCC Dataset — Sampling Report",
        f"\n_Generated on {date.today().isoformat()}_\n",
        f"**Target sample size:** {target:,}  ",
        f"**Actual downloaded:** {actual_total:,}  ",
        f"**Source grand total:** {grand_total:,}  ",
        f"**Random offset:** {RANDOM_OFFSET}\n",
        "## Per-State Allocation\n",
        "| # | State | API Total | Share % | Sample Target | Downloaded | Start Offset |",
        "|---|-------|-----------|---------|---------------|------------|--------------|",
    ]
    for i, r in enumerate(rows, 1):
        api = f"{r['api_total']:,}" if r["api_total"] else "N/A"
        pct = f"{r['pct_of_total']:.2f}%"
        target_n = f"{r['sample_target']:,}"
        dl = f"{r['sample_downloaded']:,}"
        off = f"{r['start_offset']:,}"
        lines.append(f"| {i} | {r['state']} | {api} | {pct} | {target_n} | {dl} | {off} |")

    lines.append(f"\n**Total downloaded: {actual_total:,}**\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    session = make_session()

    # --- Phase 1: probe all state totals ---
    print(f"Phase 1: Probing totals for {len(STATES)} states...\n")
    totals = {}
    fields_map = {}
    for i, state in enumerate(STATES, 1):
        print(f"  [{i:02d}/{len(STATES)}] {state}...", end=" ", flush=True)
        total, fields = fetch_total(session, state)
        totals[state] = total or 0
        fields_map[state] = fields
        print(f"total={totals[state]:,}" if totals[state] else "total=0")
        time.sleep(random.uniform(0.3, 0.7))

    grand_total = sum(totals.values())
    print(f"\nGrand total across all states: {grand_total:,}")

    # --- Phase 2: allocate sample sizes ---
    allocation = allocate_samples(totals, TARGET_SAMPLE_SIZE)
    actual_target = sum(allocation.values())

    print(f"\nPhase 2: Sample allocation (target={TARGET_SAMPLE_SIZE:,}):")
    print(f"  {'State':<25} {'API Total':>12}  {'Share':>7}  {'Sample':>8}")
    print("  " + "-" * 60)
    for state in STATES:
        t = totals[state]
        pct = (t / grand_total * 100) if grand_total else 0
        print(f"  {state:<25} {t:>12,}  {pct:>6.2f}%  {allocation[state]:>8,}")
    print(f"\n  Allocated total: {actual_target:,} records\n")

    # Clear combined file before starting
    if os.path.exists(COMBINED_FILE):
        os.remove(COMBINED_FILE)

    # --- Phase 3: download samples ---
    print("Phase 3: Downloading samples...\n")
    summary = []

    for i, state in enumerate(STATES, 1):
        sample_size = allocation[state]
        state_total = totals[state]
        pct = (state_total / grand_total * 100) if grand_total else 0

        print(f"[{i:02d}/{len(STATES)}] {state}  —  sample={sample_size:,} / total={state_total:,}")

        if sample_size == 0 or state_total == 0:
            print("  Skipping (0 records or 0 allocation).")
            summary.append({
                "state": state, "api_total": state_total, "pct_of_total": pct,
                "sample_target": 0, "sample_downloaded": 0,
                "start_offset": 0, "fields": fields_map.get(state),
            })
            save_reports(summary, grand_total, TARGET_SAMPLE_SIZE)
            continue

        # Pick a random starting offset so the sample isn't always from the top
        if RANDOM_OFFSET:
            max_start = max(0, state_total - sample_size)
            start_offset = random.randint(0, max_start)
        else:
            start_offset = 0

        print(f"  Start offset: {start_offset:,}  (random={RANDOM_OFFSET})")

        downloaded = download_sample(session, state, state_total, sample_size, start_offset)
        print(f"  Done: {downloaded:,} records written to {state_to_sample_filename(state)}")

        summary.append({
            "state": state, "api_total": state_total, "pct_of_total": pct,
            "sample_target": sample_size, "sample_downloaded": downloaded,
            "start_offset": start_offset, "fields": fields_map.get(state),
        })

        # Incremental report after each state
        save_reports(summary, grand_total, TARGET_SAMPLE_SIZE)

        time.sleep(random.uniform(0.8, 1.5))

    total_dl = sum(r["sample_downloaded"] for r in summary)
    print(f"\nAll done.")
    print(f"  Records downloaded:  {total_dl:,}")
    print(f"  Combined file:       {COMBINED_FILE}")
    print(f"  Reports:             {REPORT_CSV}, {REPORT_JSON}, {REPORT_MD}")


if __name__ == "__main__":
    main()
