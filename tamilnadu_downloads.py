#!/usr/bin/env python3
"""
tamilnadu_downloads.py

Downloads 5000 KCC records where StateName == TAMILNADU using random offsets.
Each API call uses a different random offset to sample broadly across the dataset.
"""

import os
import time
import json
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from typing import Optional, Dict, Any

# ---------- CONFIG ----------
RESOURCE_ID = "cef25fe2-9231-4128-8aec-2c948fedd43f"
API_ENDPOINT = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
API_KEY = os.getenv("KCC_API_KEY", "579b464db66ec23bdd000001e512631e0be6419550aad2c1f612a8e5")

STATE_FILTER = "TAMILNADU"   # no space — correct spelling as seen in the data
PAGE_LIMIT = 100             # records per call
TARGET_RECORDS = 5000
OUTPUT_JSONL = "kcc_data_tamilnadu.jsonl"

REQUEST_TIMEOUT = (5, 90)

RETRY_TOTAL = 8
RETRY_BACKOFF_FACTOR = 1.0
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)

MAX_ATTEMPTS_PER_PAGE = 6
BASE_PAUSE = 0.5

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "kcc-dataset-downloader/1.2",
    "Connection": "keep-alive",
}
# ----------------------------


def build_params(offset: int, limit: int) -> Dict[str, Any]:
    return {
        "api-key": API_KEY,
        "format": "json",
        "offset": offset,
        "limit": limit,
        "filters[StateName]": STATE_FILTER,
    }


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def detect_total(session: requests.Session) -> Optional[int]:
    params = build_params(offset=0, limit=1)
    try:
        r = session.get(API_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, dict):
                if "total" in j and isinstance(j["total"], int):
                    return j["total"]
                if "count" in j and isinstance(j["count"], int):
                    return j["count"]
                if "result" in j and isinstance(j["result"], dict):
                    t = j["result"].get("total")
                    if t is not None:
                        return int(t)
        else:
            print(f"detect_total: non-200 {r.status_code}")
    except Exception as e:
        print(f"detect_total: exception {e}")
    return None


def extract_records(data) -> list:
    if isinstance(data, dict):
        for key in ("records", "data", "items", "results", "response"):
            if key in data:
                return data[key]
        if "result" in data and isinstance(data["result"], dict):
            for key in ("records", "data", "items", "results"):
                if key in data["result"]:
                    return data["result"][key]
        return [data] if data else []
    elif isinstance(data, list):
        return data
    return []


def count_existing(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))


def append_jsonl(path: str, records: list):
    with open(path, "a", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fetch_page(session: requests.Session, offset: int) -> Optional[list]:
    params = build_params(offset=offset, limit=PAGE_LIMIT)
    for attempt in range(1, MAX_ATTEMPTS_PER_PAGE + 1):
        try:
            resp = session.get(API_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as e:
                    wait = (2 ** attempt) + random.random()
                    print(f"  JSON decode error (attempt {attempt}). Wait {wait:.1f}s. {e}")
                    time.sleep(wait)
                    continue
                return extract_records(data) or []
            elif resp.status_code in (429, 503, 504):
                wait = (2 ** attempt) + random.random()
                print(f"  HTTP {resp.status_code} (attempt {attempt}). Backing off {wait:.1f}s.")
                time.sleep(wait)
            elif 400 <= resp.status_code < 500:
                print(f"  Client error {resp.status_code}: {resp.text[:200]}")
                return None
            else:
                wait = (2 ** attempt) + random.random()
                print(f"  Server error {resp.status_code} (attempt {attempt}). Wait {wait:.1f}s.")
                time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = (2 ** attempt) + random.random()
            print(f"  Timeout (attempt {attempt}). Wait {wait:.1f}s.")
            time.sleep(wait)
        except requests.exceptions.RequestException as e:
            wait = (2 ** attempt) + random.random()
            print(f"  RequestException: {e} (attempt {attempt}). Wait {wait:.1f}s.")
            time.sleep(wait)
    print(f"  Gave up on offset {offset} after {MAX_ATTEMPTS_PER_PAGE} attempts.")
    return None


def download_tamilnadu():
    session = make_session()

    # Detect total TAMILNADU records so we can generate valid random offsets
    print("Detecting total TAMILNADU records...")
    total = detect_total(session)
    if total:
        print(f"Total TAMILNADU records: {total}")
    else:
        # Conservative fallback — TAMILNADU has ~200k+ records
        total = 300_000
        print(f"Could not detect total; using fallback upper bound: {total}")

    already_have = count_existing(OUTPUT_JSONL)
    if already_have >= TARGET_RECORDS:
        print(f"Already have {already_have} records in {OUTPUT_JSONL}. Nothing to do.")
        return

    print(f"Already have {already_have} records. Need {TARGET_RECORDS - already_have} more.")

    # Max valid offset: total minus one full page so we always get records
    max_offset = max(0, total - PAGE_LIMIT)

    collected = already_have
    call_count = 0
    failed_calls = 0

    while collected < TARGET_RECORDS:
        # Random offset for each call
        offset = random.randint(0, max_offset)
        call_count += 1

        needed = TARGET_RECORDS - collected
        print(f"[{collected}/{TARGET_RECORDS}] Call #{call_count} — offset={offset} (need {needed} more)")

        records = fetch_page(session, offset)

        if records is None:
            failed_calls += 1
            print(f"  Page fetch failed (total failures: {failed_calls}).")
            if failed_calls >= 10:
                print("Too many consecutive failures. Stopping.")
                break
            continue

        if not records:
            print(f"  Empty page at offset {offset}. Skipping.")
            continue

        failed_calls = 0  # reset on success

        # Trim to avoid overshooting TARGET_RECORDS
        take = min(len(records), needed)
        append_jsonl(OUTPUT_JSONL, records[:take])
        collected += take
        print(f"  Got {len(records)} records, kept {take}. Total now: {collected}")

        time.sleep(random.uniform(BASE_PAUSE, BASE_PAUSE * 2.0))

    print(f"\nDone. {collected} records written to {OUTPUT_JSONL}.")


if __name__ == "__main__":
    download_tamilnadu()
