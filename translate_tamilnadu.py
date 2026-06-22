#!/usr/bin/env python3
"""
translate_tamilnadu.py

Reads kcc_data_tamilnadu_crops.jsonl (10,000 records), picks 1500 random
English KccAns values, translates them to Tamil using Google Cloud Translate,
and writes all 10,000 records to kcc_data_tamilnadu_translated.jsonl.

Translated records get:
  - KccAns     = Tamil translation
  - KccAns_eng = original English text (new field)

Untranslated records pass through unchanged.

Usage:
  set GOOGLE_TRANSLATE_API_KEY=<your-key>
  python translate_tamilnadu.py
"""

import os
import json
import time
import html
import random
import requests
from typing import Optional

# ---------- CONFIG ----------
INPUT_JSONL  = "kcc_data_tamilnadu_crops.jsonl"
OUTPUT_JSONL = "kcc_data_tamilnadu_translated.jsonl"
TARGET_COUNT = 1500
TARGET_LANG  = "ta"
RANDOM_SEED  = 42
PAUSE_MIN    = 0.1
PAUSE_MAX    = 0.3
MAX_RETRIES  = 6
GOOGLE_TRANSLATE_API_KEY="AIzaSyDiY5YdLAAO_5nxGZGDOmGpwIbgbhtDSpY"
API_KEY="AIzaSyDiY5YdLAAO_5nxGZGDOmGpwIbgbhtDSpY"
# API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
# ----------------------------


def is_english(text: str) -> bool:
    if not text or not text.strip():
        return False
    try:
        from langdetect import detect, LangDetectException
        try:
            return detect(text) == "en"
        except LangDetectException:
            pass
    except ImportError:
        pass
    # Fallback: >=80% ASCII printable characters → treat as English
    printable = sum(1 for c in text if 32 <= ord(c) < 127)
    return (printable / len(text)) >= 0.8


def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))


TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"


def translate_text(text: str) -> Optional[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                TRANSLATE_URL,
                params={"key": API_KEY},
                json={"q": text, "target": TARGET_LANG, "format": "text"},
                timeout=(5, 30),
            )
            resp.raise_for_status()
            translated = resp.json()["data"]["translations"][0]["translatedText"]
            return html.unescape(translated)
        except Exception as e:
            wait = (2 ** attempt) + random.random()
            print(f"  Translate error (attempt {attempt}): {e}. Retrying in {wait:.1f}s.")
            time.sleep(wait)
    print(f"  Gave up translating after {MAX_RETRIES} attempts.")
    return None


def main():
    if not API_KEY:
        raise SystemExit("Set GOOGLE_TRANSLATE_API_KEY before running.")

    print(f"Using REST endpoint: {TRANSLATE_URL}")

    # Load all records
    print(f"Loading {INPUT_JSONL}...")
    with open(INPUT_JSONL, encoding="utf8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(all_records)} records.")

    # Find English indices
    print("Detecting English records...")
    english_indices = [i for i, r in enumerate(all_records) if is_english(r.get("KccAns", ""))]
    print(f"Found {len(english_indices)} English records.")

    if len(english_indices) < TARGET_COUNT:
        print(f"Warning: only {len(english_indices)} English records found; translating all of them.")
        target = len(english_indices)
    else:
        target = TARGET_COUNT

    # Pick 1500 random indices (deterministic)
    rng = random.Random(RANDOM_SEED)
    translate_set = set(rng.sample(english_indices, target))
    print(f"Selected {len(translate_set)} records for translation.")

    # Resume: skip records already written to the output
    already_written = count_lines(OUTPUT_JSONL)
    if already_written > 0:
        print(f"Resuming: {already_written} records already in {OUTPUT_JSONL}, skipping them.")
    else:
        print("Starting fresh.")

    translated_count = 0
    failed_count = 0

    for i, record in enumerate(all_records):
        if i < already_written:
            # Count how many translated records are in the already-written portion
            if i in translate_set:
                translated_count += 1
            continue

        if i in translate_set:
            tamil = translate_text(record.get("KccAns", ""))
            if tamil:
                record = dict(record)           # don't mutate the original
                record["KccAns_eng"] = record["KccAns"]
                record["KccAns"] = tamil
                translated_count += 1
                print(f"[{translated_count}/{target}] KCCCallID={record.get('KCCCallID', '?')}")
            else:
                failed_count += 1
                print(f"  Skipping KCCCallID={record.get('KCCCallID', '?')} (translation failed).")

            time.sleep(random.uniform(PAUSE_MIN, PAUSE_MAX))

        append_jsonl(OUTPUT_JSONL, record)

    total_lines = count_lines(OUTPUT_JSONL)
    print(f"\nDone.")
    print(f"  Output file   : {OUTPUT_JSONL}")
    print(f"  Total records : {total_lines}")
    print(f"  Translated    : {translated_count}")
    print(f"  Failed        : {failed_count}")


if __name__ == "__main__":
    main()
