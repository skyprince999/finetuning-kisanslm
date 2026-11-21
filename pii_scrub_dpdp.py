#!/usr/bin/env python
"""
PII scrubbing + DPDP (India's Digital Personal Data Protection Act, 2023) guardrails
for KCC (Kisan Call Centre) JSONL data, built on Microsoft Presidio.

What it does
------------
1. DETECT   Presidio AnalyzerEngine (spaCy NER + regex/checksum recognizers), extended
            with India-specific recognizers: Aadhaar (Verhoeff checksum), PAN, Voter ID,
            Passport, Vehicle Reg, GSTIN, Indian mobile/landline, bank account (context
            gated), IFSC, UPI ID, PIN code (context gated), plus honorific-name patterns
            (e.g. "Dr. Anil Verma", "MR RAVI KUMAR SINGH", "श्री ...").
2. SCRUB    Presidio AnonymizerEngine with a per-entity policy. Modes:
              redact        -> "<PHONE_NUMBER>"             (default, best for LLM training)
              mask          -> "XXXX-XXXX-1234", "98XXXXXX81"
              pseudonymize  -> "<PERSON_3f9a1c2e>"           (salted HMAC, stable per value)
3. GUARD    DPDP-oriented guardrails:
              - data minimisation: only free-text fields are scanned; output carries no raw PII
              - risk tiering per record (critical / high / medium / low / none)
              - optional quarantine of records that contained critical identifiers
              - allow-list for public helplines (e.g. KCC toll-free 1800-180-1551)
              - post-scrub leak verification (independent regex + checksum pass)
              - audit report with counts only (never the PII values themselves)
            A reusable `DPDPGuardrail` class also exposes `check_input()` / `check_output()`
            for use at inference time (e.g. in kisan_slm_chat.py).

Usage
-----
  python pii_scrub_dpdp.py --input ../kcc_data_apple.jsonl
  python pii_scrub_dpdp.py --input ../kcc_data_apple.jsonl --mode pseudonymize --quarantine-critical
  python pii_scrub_dpdp.py --limit 2000 --review 50        # quick trial + human-review sample
  python pii_scrub_dpdp.py --demo                          # run guardrail on built-in examples

Requirements
------------
  pip install presidio-analyzer presidio-anonymizer
  python -m spacy download en_core_web_lg      (or pass --spacy-model en_core_web_sm)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from presidio_analyzer import (
    AnalyzerEngine,
    BatchAnalyzerEngine,
    LocalRecognizer,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    InAadhaarRecognizer,
    InGstinRecognizer,
    InPanRecognizer,
    InPassportRecognizer,
    InVehicleRegistrationRecognizer,
    InVoterRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

TEXT_FIELDS_DEFAULT = ["QueryText", "KccAns"]

# --------------------------------------------------------------------------------------
# DPDP policy: entity -> risk tier.
#   critical : government / financial identifiers -- misuse causes direct harm
#   high     : direct contact identifiers of an individual
#   medium   : identifies a person in combination with other data
#   low      : quasi-identifiers
# LOCATION / DATE_TIME / NRP are deliberately NOT scrubbed: district/block/season are
# core agronomic context, not personal data about an identifiable individual.
# --------------------------------------------------------------------------------------
ENTITY_RISK = {
    "IN_AADHAAR": "critical",
    "IN_PAN": "critical",
    "IN_PASSPORT": "critical",
    "IN_BANK_ACCOUNT": "critical",
    "CREDIT_CARD": "critical",
    "IBAN_CODE": "critical",
    "IN_VOTER": "high",
    "UPI_ID": "high",
    "PHONE_NUMBER": "high",
    "EMAIL_ADDRESS": "high",
    "PERSON": "medium",
    "IN_VEHICLE_REGISTRATION": "medium",
    "IN_GSTIN": "medium",
    "IN_IFSC": "low",
    "IN_PINCODE": "low",
    # IP_ADDRESS intentionally omitted: NPK ratios ("0.0.50", "00:00:50") match IP patterns
}
RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

# Public, institutional numbers that are NOT personal data and are useful to keep.
DEFAULT_ALLOW_LIST = [
    r"1800[\s-]?180[\s-]?1551",  # Kisan Call Centre toll-free
    r"1800[\s-]?\d{3}[\s-]?\d{4}",  # any toll-free number
    r"\b1551\b",                  # KCC short code
    r"\b155261\b|\b1930\b|\b112\b",  # national helplines
    r"[\w.+-]+@(?:[\w-]+\.)*(?:gov|nic)\.in",  # government institutional mailboxes
]

# Agronomy vocabulary spaCy sometimes tags as PERSON (chemicals, varieties, pests).
AGRI_SUFFIXES = (
    "zole", "zeb", "phos", "thrin", "mectin", "cide", "fen", "ram", "nil", "zin",
    "mycin", "carb", "fos", "dim", "zim", "ate", "ide", "ine", "one", "xam", "lan",
)
AGRI_TERMS = {
    "mancozeb", "captan", "ziram", "propineb", "dodine", "hexaconazole", "carbendazim",
    "myclobutanil", "metiram", "pyraclostrobin", "chlorpyriphos", "imidacloprid",
    "thiamethoxam", "fenazaquin", "spiromesifen", "hexythiazox", "dimethoate", "bordeaux",
    "urea", "dap", "npk", "mop", "boron", "zinc", "calcium", "potash", "biozyme",
    "planofix", "royal", "delicious", "gala", "fuji", "ambri", "golden", "red", "chief",
    "scab", "marssonina", "alternaria", "canker", "aphid", "mites", "kvk", "kcc",
    "spray", "apple", "walnut", "petal", "custard", "horticulture", "officer",
    "farmer", "farmers", "kisan", "sir", "madam", "ji", "expert", "scientist",
    "max", "min", "price", "weather", "snow", "rain", "rainfall", "fertilizer", "dose",
    "nitrogen", "tehsil", "distt", "district", "block", "mandi", "zineb", "dithane",
    "blitox", "antracol", "sandovit", "microvit", "cabrio", "top", "flyspeck", "thrips",
}
# Words that mark a span as an institution/place rather than a person
INSTITUTION_WORDS = {
    "university", "college", "institute", "kendra", "vigyan", "krishi", "hospital",
    "marg", "road", "nagar", "colony", "department", "dept", "centre", "center", "station",
    "school", "office", "bank", "board", "mission", "society", "ltd", "limited", "pvt",
    "yojana", "scheme", "mantri", "pradhan",
    "विश्वविद्यालय", "बागवानी", "औद्यानिकी", "संस्थान", "केंद्र", "केन्द्र", "महाविद्यालय", "योजना",
}
INSTITUTION_AFTER = re.compile(
    r"(?i)^[\w.\sऀ-ॿ]{0,40}?\b(?:university|college|institute|kendra|vigyan|"
    r"hospital|yojana|scheme)\b|^[^\n]{0,40}?(?:विश्वविद्यालय|बागवानी|औद्यानिक|संस्थान|"
    r"महाविद्यालय|योजना|केंद्र|केन्द्र)")
# A real name in KCC answers is almost always introduced by one of these cues
CONTACT_CUE = re.compile(
    r"(?i)(?:contact(?:ed)?(?:\s+to)?|call|consult|meet|talk\s+to|speak\s+to|name\s*(?:is|:)?|"
    r"officer|scientist|expert|in-?charge|hdo|ado|sdo|sms|incharge|"
    r"संपर्क\s*करें|सम्पर्क\s*करें|बात\s*करें)\W{0,6}$")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


# ======================================================================================
# Custom India recognizers
# ======================================================================================
class LabelledNameRecognizer(LocalRecognizer):
    """Names that follow a form label, e.g. PM-KISAN status dumps pasted into answers:
    'Farmer Name : SURESH PATIL Father Name: ...'. Only the value is returned."""

    LABEL_RE = re.compile(
        r"(?i)\b(?:farmer'?s?|father'?s?|husband'?s?|beneficiary|applicant|account\s*holder|"
        r"guardian|mother'?s?)\s*name\s*[:\-=]\s*"
        r"|(?:किसान|पिता|पति|लाभार्थी|आवेदक)\s*(?:का|के)?\s*नाम\s*[:\-]\s*")
    VALUE_RE = re.compile(
        r"(?:[A-Za-z][A-Za-z.']*|[ऀ-ॿ]+)"
        r"(?:[ \t]+(?!(?i:father|mother|husband|mobile|aadh?aa?r|account|gender|village|address|"
        r"district|state|ifsc|bank|status|name|dob|age)\b)(?:[A-Za-z][A-Za-z.']*|[ऀ-ॿ]+)){0,4}")

    def __init__(self):
        super().__init__(supported_entities=["PERSON"], name="LabelledNameRecognizer",
                         supported_language="en")

    def load(self):
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        out = []
        for lab in self.LABEL_RE.finditer(text):
            m = self.VALUE_RE.match(text, lab.end())
            if not m or m.group().strip().upper() in {"NA", "N/A", "NULL", "NONE"}:
                continue
            out.append(RecognizerResult(
                "PERSON", m.start(), m.end(), 0.85,
                recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                                      RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id}))
        return out


def india_recognizers() -> list:
    ctx_phone = ["mobile", "phone", "contact", "call", "whatsapp", "number", "no", "mob",
                 "ph", "helpline", "मोबाइल", "फोन", "फ़ोन", "नंबर", "संपर्क"]
    recs = [
        # Built-in Presidio India recognizers (not loaded by default for "en")
        InAadhaarRecognizer(),
        InPanRecognizer(),
        InPassportRecognizer(),
        InVoterRecognizer(),
        InVehicleRegistrationRecognizer(),
        InGstinRecognizer(),
        # NOTE: Presidio's libphonenumber PhoneRecognizer is NOT used - in KCC market-price
        # tables it flags price triplets ("2000 3000 2500") as phone numbers.
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            name="InMobileRecognizer",
            patterns=[
                Pattern("in_mobile", r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)", 0.7),
                Pattern("in_landline", r"(?<!\d)0\d{2,4}[\s-]\d{5,8}(?!\d)", 0.4),
            ],
            context=ctx_phone,
        ),
        PatternRecognizer(
            supported_entity="IN_BANK_ACCOUNT",
            name="InBankAccountRecognizer",
            # very weak alone; only survives the threshold when context words are near
            patterns=[Pattern("bank_acct", r"(?<!\d)\d{9,18}(?!\d)", 0.1)],
            context=["account", "a/c", "acct", "acc", "bank", "khata", "खाता", "बैंक", "kcc"],
        ),
        PatternRecognizer(
            supported_entity="IN_IFSC",
            name="InIfscRecognizer",
            patterns=[Pattern("ifsc", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.6)],
            context=["ifsc", "branch", "bank"],
            global_regex_flags=re.MULTILINE,  # case-sensitive
        ),
        PatternRecognizer(
            supported_entity="UPI_ID",
            name="UpiRecognizer",
            patterns=[Pattern(
                "upi",
                r"\b[\w.\-]{2,64}@(?:upi|ybl|ibl|axl|apl|paytm|okaxis|oksbi|okhdfcbank|"
                r"okicici|sbi|icici|hdfcbank|axisbank|kotak|pnb|boi|barodampay|"
                r"airtel|jio|fbl|yesbank|freecharge)\b",
                0.8)],
            context=["upi", "gpay", "phonepe", "paytm", "bhim", "pay"],
        ),
        PatternRecognizer(
            supported_entity="IN_PINCODE",
            name="InPincodeRecognizer",
            patterns=[Pattern("pincode", r"(?<!\d)[1-9]\d{2}\s?\d{3}(?!\d)", 0.05)],
            context=["pin", "pincode", "pin code", "postal", "पिन"],
        ),
        # Honorific + capitalised name: catches ALL-CAPS names spaCy misses.
        PatternRecognizer(
            supported_entity="PERSON",
            name="HonorificNameRecognizer",
            patterns=[
                Pattern(
                    "en_honorific",
                    r"\b(?i:dr|mr|mrs|ms|shri|smt|sh|er|prof|sri)\.?\s+"
                    r"(?:[A-Z]\.\s*){0,3}[A-Z][A-Za-z]{1,}(?:[ \t]+(?:[A-Z]\.\s*)?[A-Z][A-Za-z]{1,}){0,3}",
                    0.65,
                ),
                Pattern(
                    "hi_honorific",
                    # honorific must start a word; "श्री" needs a space (excludes "श्रीमान" = Sir)
                    r"(?<![ऀ-ॿ])(?:श्रीमती\s+|श्री\s+|डॉ\.?\s*|डा\.\s*)"
                    r"(?!जी\b|मान\b)[ऀ-ॿ]{2,}(?:\s(?!जी\b)[ऀ-ॿ]{2,})?",
                    0.5,
                ),
            ],
            global_regex_flags=re.MULTILINE,  # case-sensitive for the name part
        ),
        LabelledNameRecognizer(),
    ]
    return recs


# ======================================================================================
# Guardrail engine
# ======================================================================================
@dataclass
class Finding:
    entity: str
    start: int
    end: int
    score: float
    risk: str

    def to_dict(self):
        return {"entity": self.entity, "start": self.start, "end": self.end,
                "score": round(self.score, 3), "risk": self.risk}


@dataclass
class ScrubResult:
    text: str
    findings: list = field(default_factory=list)

    @property
    def risk(self) -> str:
        if not self.findings:
            return "none"
        return max((f.risk for f in self.findings), key=RISK_ORDER.get)

    @property
    def entity_counts(self) -> dict:
        return dict(Counter(f.entity for f in self.findings))


@dataclass
class GuardDecision:
    allowed: bool
    text: str
    risk: str
    reasons: list


class DPDPGuardrail:
    def __init__(self, mode: str = "redact", score_threshold: float = 0.4,
                 entities: list | None = None, spacy_model: str = "en_core_web_lg",
                 allow_list: list | None = None, salt: str | None = None,
                 person_recall: str = "strict", non_person_terms: set | None = None):
        """
        person_recall: "strict" (default) - spaCy PERSON hits are kept only with a contact cue
                       before them or as a 2+ word capitalised name; best precision for
                       training data. "high" - keep every spaCy PERSON (more false positives).
        non_person_terms: extra lowercase tokens never treated as names (e.g. districts/blocks).
        """
        if mode not in {"redact", "mask", "pseudonymize"}:
            raise ValueError("mode must be redact | mask | pseudonymize")
        if person_recall not in {"strict", "high"}:
            raise ValueError("person_recall must be strict | high")
        self.mode = mode
        self.person_recall = person_recall
        self.non_person = AGRI_TERMS | {t.lower() for t in (non_person_terms or set())}
        self.threshold = score_threshold
        self.entities = entities or list(ENTITY_RISK)
        self.allow_list = DEFAULT_ALLOW_LIST if allow_list is None else allow_list
        self._allow_re = re.compile("|".join(f"(?:{p})" for p in self.allow_list)) if self.allow_list else None
        self.salt = (salt or os.environ.get("DPDP_SALT") or secrets.token_hex(16)).encode()

        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }).create_engine()
        registry = RecognizerRegistry(supported_languages=["en"])
        registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])
        registry.remove_recognizer("PhoneRecognizer")  # replaced by India-format regexes
        for r in india_recognizers():
            registry.add_recognizer(r)
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry,
                                       supported_languages=["en"])
        self.batch = BatchAnalyzerEngine(analyzer_engine=self.analyzer)
        self.anonymizer = AnonymizerEngine()
        self.operators = self._build_operators()

    # ---------------------------------------------------------------- operators
    def _pseudo(self, entity: str):
        def f(value: str) -> str:
            h = hmac.new(self.salt, value.strip().lower().encode(), hashlib.sha256).hexdigest()[:8]
            return f"<{entity}_{h}>"
        return f

    @staticmethod
    def _mask_keep_last(n: int, char: str = "X"):
        def f(value: str) -> str:
            return re.sub(r"[A-Za-z0-9]", char, value[:-n]) + value[-n:] if len(value) > n else char * len(value)
        return f

    @staticmethod
    def _mask_phone(value: str) -> str:
        digits = re.sub(r"\D", "", value)
        return digits[:2] + "X" * max(len(digits) - 4, 0) + digits[-2:] if len(digits) > 4 else "X" * len(digits)

    def _build_operators(self) -> dict:
        ops = {}
        for ent in ENTITY_RISK:
            if self.mode == "redact":
                ops[ent] = OperatorConfig("replace", {"new_value": f"<{ent}>"})
            elif self.mode == "pseudonymize":
                ops[ent] = OperatorConfig("custom", {"lambda": self._pseudo(ent)})
            else:  # mask
                if ent == "PHONE_NUMBER":
                    ops[ent] = OperatorConfig("custom", {"lambda": self._mask_phone})
                elif ent in {"IN_AADHAAR", "IN_BANK_ACCOUNT", "CREDIT_CARD", "IBAN_CODE", "IN_PAN",
                             "IN_PASSPORT", "IN_VOTER"}:
                    ops[ent] = OperatorConfig("custom", {"lambda": self._mask_keep_last(4)})
                else:  # names, emails etc. are not meaningfully maskable -> redact
                    ops[ent] = OperatorConfig("replace", {"new_value": f"<{ent}>"})
        ops["DEFAULT"] = OperatorConfig("replace", {"new_value": "<PII>"})
        return ops

    # ---------------------------------------------------------------- filtering
    def _keep(self, text: str, r: RecognizerResult) -> bool:
        span = text[r.start:r.end]
        if r.score < self.threshold:
            return False
        if self._allow_re and self._allow_re.fullmatch(span.strip()):
            return False
        if r.entity_type == "PERSON" and not self._keep_person(text, r, span):
            return False
        if r.entity_type == "EMAIL_ADDRESS" and re.search(r"@\d", span):  # "mancozeb@2.5g/l"
            return False
        if r.entity_type == "IN_AADHAAR" and not span.isdigit() and \
                not AADHAAR_CTX.search(text[max(0, r.start - 80):r.start]):
            return False  # spaced groups w/o context = market price triplets ("2500 7500 4688")
        return True

    def _keep_person(self, text: str, r: RecognizerResult, span: str) -> bool:
        from_spacy = r.recognition_metadata.get("recognizer_name") == "SpacyRecognizer"
        low = span.lower().strip(" .,:;")
        toks = re.findall(r"[^\W\d_]+", low)  # unicode letters (Latin + Devanagari)
        if not toks or len(span.strip()) < 3 or any(ch.isdigit() for ch in span) \
                or "@" in span or "/" in span:
            return False
        rec_name = r.recognition_metadata.get("recognizer_name")
        if rec_name == "LabelledNameRecognizer":
            return True  # explicit "Farmer Name: X" form field
        # "Dr. Y. S. Parmar University", "डॉ. वाई.एस. परमार बागवानी विश्वविद्यालय"
        if any(t in INSTITUTION_WORDS for t in toks) or INSTITUTION_AFTER.search(text[r.end:r.end + 60]):
            return False
        honorifics = {"dr", "mr", "mrs", "ms", "shri", "smt", "sh", "er", "prof", "sri"}
        name_toks = [t for t in toks if t not in honorifics]
        if not name_toks or all(t in self.non_person for t in name_toks):
            return False
        if not from_spacy:
            return True  # honorific pattern: high precision already
        # ---- spaCy NER hits ----
        if DEVANAGARI_RE.search(span):  # English model is unreliable on Devanagari
            return False
        if any(t in self.non_person for t in name_toks):
            return False
        if len(name_toks) == 1 and name_toks[0].endswith(AGRI_SUFFIXES):
            return False
        if self.person_recall == "high":
            return True
        # strict: require a contact cue right before the name ("contact Ramesh Kumar").
        # Title-case multi-word spans alone are too often disease/stage/scheme names.
        return bool(CONTACT_CUE.search(text[max(0, r.start - 40):r.start]))

    def _filter(self, text: str, results: list) -> list:
        return [r for r in results if self._keep(text, r)]

    # ---------------------------------------------------------------- core API
    def analyze(self, text: str) -> list:
        if not text:
            return []
        res = self.analyzer.analyze(text=text, language="en", entities=self.entities,
                                    score_threshold=self.threshold)
        return self._filter(text, res)

    def analyze_many(self, texts: list, batch_size: int = 256, n_process: int = 1) -> list:
        out = self.batch.analyze_iterator(texts, language="en", entities=self.entities,
                                          score_threshold=self.threshold,
                                          batch_size=batch_size, n_process=n_process)
        return [self._filter(t, r) if t else [] for t, r in zip(texts, out)]

    def anonymize(self, text: str, results: list) -> ScrubResult:
        if not results:
            return ScrubResult(text=text)
        anon = self.anonymizer.anonymize(text=text, analyzer_results=results, operators=self.operators)
        # findings reflect what was actually applied (anonymizer resolves overlaps)
        applied = {(i.entity_type) for i in anon.items}
        findings = [Finding(r.entity_type, r.start, r.end, r.score,
                            ENTITY_RISK.get(r.entity_type, "low"))
                    for r in _dedupe_overlaps(results) if r.entity_type in applied]
        return ScrubResult(text=anon.text, findings=findings)

    def scrub(self, text: str) -> ScrubResult:
        return self.anonymize(text, self.analyze(text))

    # ---------------------------------------------------------------- inference guardrails
    def check_input(self, user_text: str) -> GuardDecision:
        """Scrub a farmer's prompt before it is logged / sent to a model.
        Input is always allowed (the farmer may legitimately share a number); PII is removed."""
        r = self.scrub(user_text)
        reasons = [f"{k} x{v} scrubbed" for k, v in r.entity_counts.items()]
        return GuardDecision(True, r.text, r.risk, reasons)

    def check_output(self, model_text: str, block_at: str = "critical") -> GuardDecision:
        """Guard a model response: block if it leaks identifiers at/above `block_at`,
        otherwise return a scrubbed version (prevents memorised phone numbers/names leaking)."""
        r = self.scrub(model_text)
        reasons = [f"{k} x{v}" for k, v in r.entity_counts.items()]
        if RISK_ORDER[r.risk] >= RISK_ORDER[block_at]:
            return GuardDecision(False, "क्षमा करें, इस उत्तर में व्यक्तिगत जानकारी थी इसलिए इसे रोका गया। / "
                                        "Sorry, this response was withheld because it contained personal data.",
                                 r.risk, reasons)
        return GuardDecision(True, r.text, r.risk, reasons)


def _dedupe_overlaps(results: list) -> list:
    """Keep highest-score result among overlapping spans (mirrors anonymizer conflict handling)."""
    out = []
    for r in sorted(results, key=lambda x: (-x.score, x.start)):
        if all(r.end <= o.start or r.start >= o.end for o in out):
            out.append(r)
    return sorted(out, key=lambda x: x.start)


# ======================================================================================
# Independent post-scrub leak verification (does not reuse Presidio)
# ======================================================================================
_VERHOEFF_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
               [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
               [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
               [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
               [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VERHOEFF_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
               [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
               [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
               [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_ok(num: str) -> bool:
    c = 0
    for i, d in enumerate(reversed(num)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(d)]]
    return c == 0


LEAK_CHECKS = {
    "mobile": re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    # contiguous 12 digits, or spaced/dashed groups (spaced form needs Aadhaar context,
    # otherwise price tables like "3000 7500 6000" pass the checksum ~10% of the time)
    "aadhaar": re.compile(r"(?<!\d)[2-9]\d{11}(?!\d)|(?<!\d)[2-9]\d{3}[ -]\d{4}[ -]\d{4}(?!\d)"),
    "pan": re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b"),
    "email": re.compile(r"\b[\w.+-]+@[A-Za-z][\w-]*(?:\.[\w-]+)*\.[A-Za-z]{2,}\b"),
}
AADHAAR_CTX = re.compile(r"(?i)aadh?aa?r|\buid\b|आधार")


def leak_scan(text: str, allow_re: re.Pattern | None = None) -> list:
    hits = []
    text = text or ""
    for name, rx in LEAK_CHECKS.items():
        for m in rx.finditer(text):
            if allow_re and allow_re.fullmatch(m.group().strip()):
                continue
            if name == "aadhaar":
                digits = re.sub(r"\D", "", m.group())
                if not verhoeff_ok(digits):
                    continue
                if digits != m.group() and \
                        not AADHAAR_CTX.search(text[max(0, m.start() - 60):m.start()]):
                    continue
            hits.append(name)
    return hits


# ======================================================================================
# Batch pipeline for JSONL
# ======================================================================================
def resolve_input(p: str) -> Path:
    path = Path(p)
    if path.exists():
        return path
    alt = Path(__file__).resolve().parent.parent / p
    if alt.exists():
        return alt
    sys.exit(f"Input not found: {p}")


def read_jsonl(path: Path, limit: int | None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def run_pipeline(args):
    src = resolve_input(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent / "pii_scrubbed"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    out_path = out_dir / f"{stem}_scrubbed.jsonl"
    quarantine_path = out_dir / f"{stem}_quarantine.jsonl"
    audit_path = out_dir / f"{stem}_dpdp_audit.json"
    review_path = out_dir / f"{stem}_REVIEW_CONTAINS_PII.csv"

    t0 = time.time()
    print(f"[1/5] Reading {src} ...")
    records = list(read_jsonl(src, args.limit))
    fields = args.fields
    print(f"      {len(records):,} records; fields scanned: {fields}")

    # Gazetteer from the record metadata: district/block/state names are places, not people
    places = set()
    for rec in records:
        for k in ("StateName", "DistrictName", "BlockName"):
            places.update(re.findall(r"[^\W\d_]+", str(rec.get(k) or "").lower()))
    places -= {"na", "and", "of", "the"}

    print(f"[2/5] Loading guardrail (spaCy model: {args.spacy_model}, mode: {args.mode}, "
          f"person recall: {args.person_recall}) ...")
    guard = DPDPGuardrail(mode=args.mode, score_threshold=args.threshold,
                          spacy_model=args.spacy_model,
                          allow_list=DEFAULT_ALLOW_LIST + (args.allow or []),
                          person_recall=args.person_recall, non_person_terms=places)
    if args.mode == "pseudonymize" and not os.environ.get("DPDP_SALT"):
        print("      ! DPDP_SALT not set: using a random per-run salt (pseudonyms won't be "
              "stable across runs). Set DPDP_SALT and store it separately from the data.")

    # Analyse each unique text only once (KCC data is highly repetitive)
    print("[3/5] Detecting PII on unique texts ...")
    cache: dict[str, ScrubResult] = {}
    uniq = sorted({(r.get(f) or "") for r in records for f in fields} - {""})
    print(f"      {len(uniq):,} unique non-empty texts")
    done = 0
    for i in range(0, len(uniq), args.chunk):
        chunk = uniq[i:i + args.chunk]
        for text, res in zip(chunk, guard.analyze_many(chunk, batch_size=args.batch_size,
                                                        n_process=args.n_process)):
            cache[text] = guard.anonymize(text, res)
        done += len(chunk)
        el = time.time() - t0
        print(f"      {done:,}/{len(uniq):,} texts  ({el:,.0f}s elapsed)", flush=True)

    print("[4/5] Writing scrubbed output + guardrail checks ...")
    stats = {
        "entities_by_field": defaultdict(Counter),
        "records_by_risk": Counter(),
        "records_with_pii": 0,
        "quarantined": 0,
        "residual_leaks": Counter(),
        "residual_leak_records": [],
    }
    review_rows = []
    with out_path.open("w", encoding="utf-8") as fout, \
            (quarantine_path.open("w", encoding="utf-8") if args.quarantine_critical else _Null()) as fq:
        for idx, rec in enumerate(records):
            new = dict(rec)
            rec_risk, rec_entities = "none", Counter()
            for f in fields:
                orig = rec.get(f) or ""
                if not orig:
                    continue
                sr = cache.get(orig, ScrubResult(orig))
                new[f] = sr.text
                for ent, n in sr.entity_counts.items():
                    stats["entities_by_field"][f][ent] += n
                    rec_entities[ent] += n
                if RISK_ORDER[sr.risk] > RISK_ORDER[rec_risk]:
                    rec_risk = sr.risk
                if sr.findings and len(review_rows) < args.review:
                    review_rows.append({"line": idx + 1, "field": f, "entities": sr.entity_counts,
                                        "original": orig, "scrubbed": sr.text})
                leaks = leak_scan(new[f], guard._allow_re)
                if leaks:
                    stats["residual_leaks"].update(leaks)
                    if len(stats["residual_leak_records"]) < 200:
                        stats["residual_leak_records"].append({"line": idx + 1, "field": f, "types": leaks})

            stats["records_by_risk"][rec_risk] += 1
            if rec_entities:
                stats["records_with_pii"] += 1
            if not args.no_annotate:
                new["_dpdp"] = {"risk": rec_risk, "entities": dict(rec_entities)}

            if args.quarantine_critical and rec_risk == "critical":
                stats["quarantined"] += 1
                fq.write(json.dumps(new, ensure_ascii=False) + "\n")  # scrubbed form only
                continue
            fout.write(json.dumps(new, ensure_ascii=False) + "\n")

    if args.review and review_rows:
        with review_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["line", "field", "entities", "original", "scrubbed"])
            w.writeheader()
            w.writerows(review_rows)

    print("[5/5] Writing audit report ...")
    total_ent = Counter()
    for c in stats["entities_by_field"].values():
        total_ent.update(c)
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framework": "Digital Personal Data Protection Act, 2023 (India) - engineering guardrails",
        "input_file": src.name,
        "input_sha256": _sha256(src) if not args.limit else "skipped (--limit used)",
        "output_file": out_path.name,
        "config": {
            "mode": args.mode, "score_threshold": args.threshold, "spacy_model": args.spacy_model,
            "person_recall": args.person_recall,
            "fields_scanned": fields, "entities_in_scope": guard.entities,
            "entity_risk_tiers": ENTITY_RISK, "allow_list_patterns": guard.allow_list,
            "quarantine_critical": args.quarantine_critical,
        },
        "records_total": len(records),
        "records_written": len(records) - stats["quarantined"],
        "records_quarantined": stats["quarantined"],
        "records_with_pii": stats["records_with_pii"],
        "records_by_risk": dict(stats["records_by_risk"]),
        "entities_total": dict(total_ent.most_common()),
        "entities_by_field": {k: dict(v.most_common()) for k, v in stats["entities_by_field"].items()},
        "post_scrub_verification": {
            "residual_leak_counts": dict(stats["residual_leaks"]),
            "status": "PASS" if not stats["residual_leaks"] else "REVIEW",
            "sample_locations": stats["residual_leak_records"][:50],
        },
        "dpdp_controls": {
            "data_minimisation": "Only free-text fields scanned; identifiers replaced, never copied to logs.",
            "purpose_limitation": "Scrubbed output intended for model training/analytics on agronomic advice.",
            "audit_without_pii": "This report holds counts and line numbers only - no personal data values.",
            "pseudonymisation_key": "Salt from DPDP_SALT env var; store separately from data." if args.mode == "pseudonymize" else "n/a",
            "human_review_file": review_path.name + " (CONTAINS RAW PII - restrict access, delete after review)" if args.review else "not generated",
            "known_limitations": [
                "spaCy English NER does not detect names written in Devanagari unless preceded by an honorific (श्री/श्रीमती/डॉ).",
                "Names without honorifics in ALL-CAPS text may be missed by NER.",
                "Context-gated entities (bank account, PIN code) require a nearby keyword.",
            ],
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    # console summary
    print("\n" + "=" * 70)
    print(f"Records: {len(records):,} | with PII: {stats['records_with_pii']:,} "
          f"({stats['records_with_pii'] / max(len(records), 1) * 100:.2f}%) | quarantined: {stats['quarantined']:,}")
    print("Risk tiers:", dict(stats["records_by_risk"]))
    print("Entities found:", dict(total_ent.most_common()))
    print(f"Post-scrub verification: {audit['post_scrub_verification']['status']} "
          f"{dict(stats['residual_leaks']) or ''}")
    print(f"\nOutput     : {out_path}")
    if args.quarantine_critical:
        print(f"Quarantine : {quarantine_path}")
    print(f"Audit      : {audit_path}")
    if args.review and review_rows:
        print(f"Review     : {review_path}  <-- CONTAINS RAW PII, delete after review")
    print(f"Elapsed    : {time.time() - t0:,.0f}s")

    if args.strict and stats["residual_leaks"]:
        sys.exit(2)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, *_):
        pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ======================================================================================
# Demo
# ======================================================================================
DEMO_TEXTS = [
    "For further information contact MR RAVI KUMAR SINGH 9800000001",
    "Please Contact Dr. Anil Verma, HOD Fruit Science, Example Uni. Phone 01234-567890",
    "Farmer's Aadhar Number 2345 6789 0124, account number 30211876543 IFSC SBIN0001234",
    "PM-Kisan status: PAN ABCPK1234F, pay via ramesh.k@oksbi or mail ramesh@gmail.com",
    "Farmer Application Status Farmer Name : SURESH PATIL Father Name: NA Mobile Number: NA",
    "Contact Dr. Y. S. Parmar University, Nauni for Tight Cluster stage spray of Pink Bud.",
    "Spray chlorpyriphos@20ml/10L and Mancozeb 75WP (300 g) at petal fall stage.",
    "किसान कॉल सेंटर के व्हाट्सएप नंबर 9800000002 पर तस्वीरें भेजें या 1800-180-1551 पर कॉल करें। श्री रमेश कुमार",
]


def run_demo(args):
    for mode in ("redact", "mask", "pseudonymize"):
        g = DPDPGuardrail(mode=mode, spacy_model=args.spacy_model, salt="demo-salt")
        print(f"\n===== mode: {mode} =====")
        for t in DEMO_TEXTS:
            r = g.scrub(t)
            print(f"[{r.risk:8}] {r.text}")
    g = DPDPGuardrail(spacy_model=args.spacy_model)
    print("\n===== inference output guardrail =====")
    for t in DEMO_TEXTS[:3]:
        d = g.check_output(t)
        print(f"allowed={d.allowed!s:5} risk={d.risk:8} {d.reasons} -> {d.text}")


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description="PII scrubbing + DPDP guardrails for KCC JSONL (Presidio)")
    ap.add_argument("--input", default="kcc_data_apple.jsonl")
    ap.add_argument("--out-dir", default=None, help="default: <input dir>/pii_scrubbed")
    ap.add_argument("--fields", nargs="+", default=TEXT_FIELDS_DEFAULT)
    ap.add_argument("--mode", choices=["redact", "mask", "pseudonymize"], default="redact")
    ap.add_argument("--threshold", type=float, default=0.4, help="min Presidio confidence")
    ap.add_argument("--spacy-model", default="en_core_web_lg")
    ap.add_argument("--person-recall", choices=["strict", "high"], default="strict",
                    help="strict: spaCy names need a contact cue / multi-word name (fewer false "
                         "positives on chemical & place names); high: keep all spaCy PERSON hits")
    ap.add_argument("--allow", nargs="*", help="extra regex patterns to never scrub")
    ap.add_argument("--quarantine-critical", action="store_true",
                    help="write records that contained Aadhaar/PAN/bank/card to a separate file")
    ap.add_argument("--no-annotate", action="store_true", help="don't add the _dpdp field to records")
    ap.add_argument("--review", type=int, default=0,
                    help="write N before/after rows for human QA (file contains raw PII)")
    ap.add_argument("--strict", action="store_true", help="exit 2 if post-scrub leaks are found")
    ap.add_argument("--limit", type=int, default=None, help="process only first N records")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-process", type=int, default=1, help="spaCy worker processes")
    ap.add_argument("--chunk", type=int, default=20000, help="progress-report chunk size")
    ap.add_argument("--demo", action="store_true", help="run on built-in examples and exit")
    args = ap.parse_args()

    if args.demo:
        run_demo(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
