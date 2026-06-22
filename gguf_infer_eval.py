#!/usr/bin/env python3
"""
gguf_infer_eval.py

Combined inference + evaluation pipeline for KCC fine-tune comparison.

Runs llama.cpp-based inference over `skylord/kcc-sample100k` (TEST split)
for every model listed in MODELS, in two phases:
  - baseline   : original un-finetuned instruct GGUFs (unsloth namespace)
  - finetuned  : GGUFs pushed to the skylord namespace after fine-tuning

Then scores every (reference, generated) pair with sentence-transformer
cosine similarity and writes per-row JSONL + a CSV summary so you can
compare baseline vs fine-tuned across all models in one shot.

Resume-safe: rows already present in OUTPUT_JSONL are skipped on re-run.

Known issue with qwen3-0.6b finetuned: the published checkpoint
(skylord/qwen3-0.6b-kcc-sample100k-torchao) is in torchao format, not
GGUF. That model will be skipped for the finetuned phase with a warning.

Install:
    pip install llama-cpp-python huggingface_hub datasets python-dotenv
    pip install sentence-transformers numpy
    # GPU offload (optional):
    #   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir

Auth: put HF_TOKEN=hf_... in .env
"""

import os
import gc
import json
import random
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ───────────────────────── CONFIG ──────────────────────────
DATASET_REPO     = "skylord/kcc-sample100k"
TEST_SPLIT       = "test"
MAX_TEST_SAMPLES = 50       # None = full 500-record test split

# Inference knobs
MAX_NEW_TOKENS = 4096
TEMPERATURE    = 0.2
N_CTX          = 2048
N_GPU_LAYERS   = -1         # -1 = all layers to GPU if available

# Which phases to run
RUN_BASELINE  = True
RUN_FINETUNED = True        # was False in the old script — now on by default

# Output files
OUTPUT_JSONL    = "kcc_inference_results.jsonl"
EVAL_OUTPUT     = "kcc_eval_results.jsonl"
SUMMARY_OUTPUT  = "kcc_eval_summary.csv"

# Embedding model for cosine-similarity scoring (local, no API cost)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Optional LLM-judge scoring (costs money — off by default)
RUN_GEVAL         = True
GEVAL_SAMPLE_SIZE = None    # max rows scored PER (model, phase) group — None = all rows
GEVAL_MODEL       = "gpt-4o-mini"
GEVAL_SEED        = 42

QUANT_PREFERENCE = ["q4_k_m", "q4_k_xl", "q4_k_s", "q6_k", "q8_0"]

# ── Model registry ──
# Fixes vs the old run_inference_kcc_test.py:
#   • gemma-3n finetuned_repo had "skylord/skylord/..." (double namespace) → fixed
#   • qwen3-1.7b and qwen3-0.6b baseline repos were missing -GGUF suffix → fixed
#   • qwen3-1.7b finetuned contains safetensors → loaded via transformers backend
#   • qwen3-0.6b finetuned is torchao → loaded via transformers backend (needs pip install torchao)
#
# finetuned_backend options:
#   "llama_cpp"    — download a .gguf file and run via llama-cpp-python (default)
#   "transformers" — load directly from HF hub via AutoModelForCausalLM
MODELS = [
    {
        "key":            "gemma-3n-e4b",
        "baseline_repo":  "unsloth/gemma-3n-E4B-it-GGUF",
        "finetuned_repo": "skylord/gemma-3n-kcc-sample100k-gguf-step6300",
    },
    {
        "key":            "llama-3.2-1b",
        "baseline_repo":  "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "finetuned_repo": "skylord/llama-3.2-1b-kcc-sample100k-gguf",
    },
    {
        "key":            "lfm2.5-1.2b",
        "baseline_repo":  "unsloth/LFM2.5-1.2B-Instruct-GGUF",
        "finetuned_repo": "skylord/lfm2-1.2b-kcc-sample100k-gguf",
    },
    {
        "key":               "qwen3-1.7b",
        "baseline_repo":     "unsloth/Qwen3-1.7B-GGUF",
        "finetuned_repo":    "skylord/qwen3-1.7b-kcc-gguf",
        "finetuned_backend": "transformers",   # repo has safetensors, not GGUF
    },
    # {
    #     "key":               "qwen3-0.6b",
    #     "baseline_repo":     "unsloth/Qwen3-0.6B-GGUF",
    #     "finetuned_repo":    "skylord/qwen3-0.6b-kcc-sample100k-torchao",
    #     "finetuned_backend": "transformers",   # torchao safetensors — needs: pip install torchao
    # },
]
# ───────────────────────────────────────────────────────────


# ══════════════════════════ INFERENCE ══════════════════════════

from huggingface_hub import list_repo_files, hf_hub_download
from llama_cpp import Llama


def pick_gguf_filename(repo_id: str) -> str:
    files = [f for f in list_repo_files(repo_id, token=HF_TOKEN) if f.lower().endswith(".gguf")]
    if not files:
        raise FileNotFoundError(f"No .gguf files found in {repo_id}")
    for pref in QUANT_PREFERENCE:
        matches = sorted(f for f in files if pref in f.lower())
        if matches:
            return matches[0]
    return sorted(files)[0]


def download_gguf(repo_id: str) -> str:
    filename = pick_gguf_filename(repo_id)
    print(f"  Downloading {repo_id}/{filename}")
    return hf_hub_download(repo_id=repo_id, filename=filename, token=HF_TOKEN)


def load_llm_gguf(gguf_path: str) -> Llama:
    return Llama(model_path=gguf_path, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)


def generate_gguf(llm: Llama, query: str) -> str:
    resp = llm.create_chat_completion(
        messages=[{"role": "user", "content": query}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    return resp["choices"][0]["message"]["content"]


def generate_transformers(repo_id: str, query: str, pipeline_cache: dict) -> str:
    """Load (and cache) a transformers pipeline, then generate one response."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

    if repo_id not in pipeline_cache:
        print(f"  Loading {repo_id} via transformers...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        pipeline_cache[repo_id] = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    pipe = pipeline_cache[repo_id]
    messages = [{"role": "user", "content": query}]
    out = pipe(
        messages,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=TEMPERATURE > 0,
    )
    return out[0]["generated_text"][-1]["content"]


def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_keys(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["model_key"], r["phase"], r["KCCCallID"]))
    return done


def run_inference_pass(phase: str, model_key: str, repo_id: str,
                       test_records: list, done_keys: set,
                       backend: str = "llama_cpp",
                       pipeline_cache: dict = None):
    print(f"\n=== {phase.upper()} | {model_key} | {repo_id} (backend={backend}) ===")

    # Set up llama_cpp model once before the loop
    llm = None
    if backend == "llama_cpp":
        try:
            gguf_path = download_gguf(repo_id)
        except Exception as e:
            print(f"  SKIP — could not download GGUF: {e}")
            return
        llm = load_llm_gguf(gguf_path)

    n_done = sum(1 for r in test_records if (model_key, phase, r["KCCCallID"]) in done_keys)
    if n_done:
        print(f"  Resuming: {n_done}/{len(test_records)} already done.")

    for i, rec in enumerate(test_records):
        key = (model_key, phase, rec["KCCCallID"])
        if key in done_keys:
            continue
        try:
            if backend == "llama_cpp":
                generated = generate_gguf(llm, rec["QueryText"])
            else:
                generated = generate_transformers(repo_id, rec["QueryText"], pipeline_cache)
        except Exception as e:
            generated = f"[ERROR: {e}]"

        append_jsonl(OUTPUT_JSONL, {
            "model_key":        model_key,
            "phase":            phase,
            "repo_id":          repo_id,
            "backend":          backend,
            "KCCCallID":        rec.get("KCCCallID"),
            "StateName":        rec.get("StateName"),
            "QueryText":        rec.get("QueryText"),
            "reference_answer": rec.get("KccAns"),
            "generated_answer": generated,
        })
        done_keys.add(key)
        print(f"  [{i + 1}/{len(test_records)}] KCCCallID={rec.get('KCCCallID')} done.")

    if llm is not None:
        del llm
        gc.collect()


def run_inference(test_records: list):
    done_keys = load_done_keys(OUTPUT_JSONL)
    # transformers pipelines are kept alive across rows to avoid reloading per row,
    # but released between models by clearing the cache after each pass
    pipeline_cache: dict = {}

    for m in MODELS:
        if RUN_BASELINE:
            run_inference_pass(
                "baseline", m["key"], m["baseline_repo"],
                test_records, done_keys,
                backend="llama_cpp",
                pipeline_cache=pipeline_cache,
            )
            pipeline_cache.clear()
            gc.collect()

        if RUN_FINETUNED:
            backend = m.get("finetuned_backend", "llama_cpp")
            run_inference_pass(
                "finetuned", m["key"], m["finetuned_repo"],
                test_records, done_keys,
                backend=backend,
                pipeline_cache=pipeline_cache,
            )
            pipeline_cache.clear()
            gc.collect()


# ══════════════════════════ EVALUATION ══════════════════════════

def load_inference_results(path: str) -> list:
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_cosine_similarities(rows: list) -> list:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading embedding model {EMBEDDING_MODEL} (local, free)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    references  = [r.get("reference_answer") or "" for r in rows]
    generations = [r.get("generated_answer") or "" for r in rows]

    print(f"Embedding {len(rows)} reference/generated pairs...")
    ref_emb = model.encode(references,  batch_size=64, show_progress_bar=True)
    gen_emb = model.encode(generations, batch_size=64, show_progress_bar=True)

    import numpy as np
    ref_emb = np.asarray(ref_emb)
    gen_emb = np.asarray(gen_emb)

    dot  = (ref_emb * gen_emb).sum(axis=1)
    norm = np.linalg.norm(ref_emb, axis=1) * np.linalg.norm(gen_emb, axis=1)
    sims = dot / (norm + 1e-9)
    return sims.tolist()


def run_geval_sample(rows: list) -> dict:
    if not RUN_GEVAL:
        print("\n[GEVAL SKIPPED] RUN_GEVAL=False — set RUN_GEVAL=True to enable LLM-judge scoring.")
        return {}

    if not os.getenv("OPENAI_API_KEY"):
        print("\n[GEVAL SKIPPED] OPENAI_API_KEY not found in environment or .env file.")
        print("                Add OPENAI_API_KEY=sk-... to your .env file and re-run.")
        return {}

    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, SingleTurnParams
    except ImportError:
        print("\n[GEVAL SKIPPED] deepeval is not installed — run: pip install deepeval")
        return {}

    # Group rows by (model_key, phase) and score every row in each group.
    # GEVAL_SAMPLE_SIZE caps rows PER GROUP (not globally) — set to None to score all.
    from collections import defaultdict as _dd
    groups = _dd(list)
    for row in rows:
        groups[(row["model_key"], row["phase"])].append(row)

    metric = GEval(
        name="Semantic Similarity",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        evaluation_steps=[
            "Check if the core event/fact in the expected output is present in the actual output.",
            "Ensure no contradictory facts are introduced.",
            "Ignore minor wording or syntax differences as long as the meaning is identical.",
        ],
        threshold=0.7,
        model=GEVAL_MODEL,
    )

    scores = {}
    total_groups = len(groups)
    for g_idx, ((model_key, phase), group_rows) in enumerate(sorted(groups.items()), 1):
        if GEVAL_SAMPLE_SIZE is not None and len(group_rows) > GEVAL_SAMPLE_SIZE:
            rng   = random.Random(GEVAL_SEED)
            group_rows = rng.sample(group_rows, GEVAL_SAMPLE_SIZE)
        skipped = 0
        print(f"  [{g_idx}/{total_groups}] GEval: {model_key} / {phase} — {len(group_rows)} rows...")
        for row in group_rows:
            actual   = (row.get("generated_answer") or "").strip()
            expected = (row.get("reference_answer")  or "").strip()
            if not actual or actual.startswith("[ERROR:") or not expected:
                skipped += 1
                continue
            test_case = LLMTestCase(
                input=row.get("QueryText", ""),
                actual_output=actual,
                expected_output=expected,
            )
            metric.measure(test_case)
            key = (row["model_key"], row["phase"], row["KCCCallID"])
            scores[key] = metric.score
        if skipped:
            print(f"    Skipped {skipped} rows (empty or errored output).")
    return scores


def run_evaluation():
    rows = load_inference_results(OUTPUT_JSONL)
    print(f"\nLoaded {len(rows)} inference results from {OUTPUT_JSONL}.")

    sims         = compute_cosine_similarities(rows)
    geval_scores = run_geval_sample(rows)

    with open(EVAL_OUTPUT, "w", encoding="utf8") as f:
        for row, sim in zip(rows, sims):
            key = (row["model_key"], row["phase"], row["KCCCallID"])
            f.write(json.dumps({
                "model_key":        row["model_key"],
                "phase":            row["phase"],
                "KCCCallID":        row["KCCCallID"],
                "cosine_similarity": round(float(sim), 4),
                "geval_score":       geval_scores.get(key),
            }, ensure_ascii=False) + "\n")

    # Aggregate per (model_key, phase) — cosine sim for every row, geval only for sampled rows
    agg_cos   = defaultdict(list)
    agg_geval = defaultdict(list)
    for row, sim in zip(rows, sims):
        k = (row["model_key"], row["phase"])
        agg_cos[k].append(sim)
        geval_key = (row["model_key"], row["phase"], row["KCCCallID"])
        if geval_key in geval_scores:
            agg_geval[k].append(geval_scores[geval_key])

    print("\n=== Summary (mean cosine similarity) ===")
    summary_rows = []
    for k, cos_vals in sorted(agg_cos.items()):
        model_key, phase = k
        mean_cos   = sum(cos_vals) / len(cos_vals)
        geval_vals = agg_geval.get(k, [])
        mean_geval = sum(geval_vals) / len(geval_vals) if geval_vals else None
        geval_str  = f"  mean_geval={mean_geval:.4f} (n={len(geval_vals)})" if mean_geval is not None else ""
        print(f"  {model_key:15s}  {phase:10s}  n={len(cos_vals):4d}  mean_cosine={mean_cos:.4f}{geval_str}")
        summary_rows.append((model_key, phase, len(cos_vals), mean_cos, mean_geval, len(geval_vals)))

    # Baseline vs finetuned delta per model
    by_model: dict[str, dict] = defaultdict(dict)
    for model_key, phase, n, mean_cos, mean_geval, n_geval in summary_rows:
        by_model[model_key][phase] = {"cos": mean_cos, "geval": mean_geval}

    print("\n=== Delta (finetuned − baseline) ===")
    for model_key in sorted(by_model):
        phases = by_model[model_key]
        if "baseline" in phases and "finetuned" in phases:
            delta_cos = phases["finetuned"]["cos"] - phases["baseline"]["cos"]
            sign = "+" if delta_cos >= 0 else ""
            geval_b = phases["baseline"]["geval"]
            geval_f = phases["finetuned"]["geval"]
            if geval_b is not None and geval_f is not None:
                delta_geval = geval_f - geval_b
                gsign = "+" if delta_geval >= 0 else ""
                print(f"  {model_key:15s}  cosine {sign}{delta_cos:.4f}  geval {gsign}{delta_geval:.4f}")
            else:
                print(f"  {model_key:15s}  cosine {sign}{delta_cos:.4f}  geval n/a")
        else:
            missing = {"baseline", "finetuned"} - set(phases)
            print(f"  {model_key:15s}  (missing phases: {', '.join(missing)})")

    with open(SUMMARY_OUTPUT, "w", encoding="utf8") as f:
        f.write("model_key,phase,n,mean_cosine_similarity,mean_geval_score,n_geval\n")
        for model_key, phase, n, mean_cos, mean_geval, n_geval in summary_rows:
            geval_cell = f"{mean_geval:.4f}" if mean_geval is not None else ""
            f.write(f"{model_key},{phase},{n},{mean_cos:.4f},{geval_cell},{n_geval}\n")

    print(f"\nPer-row scores : {EVAL_OUTPUT}")
    print(f"Summary table  : {SUMMARY_OUTPUT}")
    if RUN_GEVAL and geval_scores:
        print(f"GEval scores computed for {len(geval_scores)} sampled rows, aggregated into summary.")


# ══════════════════════════ MAIN ══════════════════════════

def main():
    from datasets import load_dataset

    print(f"Loading {DATASET_REPO} ({TEST_SPLIT} split)...")
    test_ds      = load_dataset(DATASET_REPO, split=TEST_SPLIT)
    test_records = list(test_ds)
    if MAX_TEST_SAMPLES is not None:
        test_records = test_records[:MAX_TEST_SAMPLES]
    print(f"Running inference on {len(test_records)} test records.")

    run_inference(test_records)

    print("\n" + "=" * 60)
    print("Inference complete. Starting evaluation...")
    print("=" * 60)

    run_evaluation()


if __name__ == "__main__":
    main()
