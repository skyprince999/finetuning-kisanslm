#!/usr/bin/env python3
"""KCC Fine-tuned Model Chat — Gradio UI"""

import gc
import os
import threading
import time
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download, list_repo_files

# ── Load .env from the same directory as this script ─────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set externally

HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# ── Model registry ────────────────────────────────────────────────────────────

GGUF_MODELS: dict[str, str] = {
    "Gemma 3N (GGUF)":     "skylord/gemma-3n-kcc-sample100k-gguf",
    "Llama 3.2 1B (GGUF)": "skylord/llama-3.2-1b-kcc-sample100k-gguf",
    "LFM2 1.2B (GGUF)":    "skylord/lfm2-1.2b-kcc-sample100k-gguf",
}

# Safetensors / HF transformers models (no GGUF available)
HF_MODELS: dict[str, str] = {
    "Qwen3 1.7B (HF)":       "skylord/qwen3-1.7b-kcc-gguf",
    "Qwen3 0.6B (TorchAO)":  "skylord/qwen3-0.6b-kcc-sample100k-torchao",
}

ALL_MODELS = list(GGUF_MODELS) + list(HF_MODELS)

# Default context window per model (tokens)
MODEL_CTX: dict[str, int] = {
    "Gemma 3N (GGUF)":      4096,
    "Llama 3.2 1B (GGUF)":  4096,
    "LFM2 1.2B (GGUF)":     4096,
    "Qwen3 1.7B (HF)":      4096,
    "Qwen3 0.6B (TorchAO)": 4096,
}

# ── Global model state ────────────────────────────────────────────────────────

_state: dict = {"model": None, "tokenizer": None, "name": None, "type": None}
_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_gguf(repo_id: str) -> str:
    all_files = list(list_repo_files(repo_id, token=HF_TOKEN))
    gguf_files = [f for f in all_files if f.lower().endswith(".gguf")]
    if not gguf_files:
        raise FileNotFoundError(
            f"No .gguf file found in {repo_id}.\nAvailable: {all_files}"
        )
    for tag in ("Q4_K_M", "q4_k_m", "Q8_0", "q8_0", "Q4_0", "q4_0"):
        for f in gguf_files:
            if tag in f:
                return f
    return gguf_files[0]


def _unload():
    _state["model"] = None
    _state["tokenizer"] = None
    _state["name"] = None
    _state["type"] = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _fmt_stats(prompt_tok: int, prefill_s: float, gen_tok: int, gen_s: float) -> str:
    p_tps = prompt_tok / prefill_s if prefill_s > 0 else 0.0
    g_tps = gen_tok / gen_s if gen_s > 0 else 0.0
    return (
        f"Prompt:  {prompt_tok} tokens | {p_tps:,.0f} tok/s  "
        f"({prefill_s*1000:.0f} ms)    "
        f"Generation:  {gen_tok} tokens | {g_tps:.1f} tok/s  "
        f"({gen_s:.1f} s)"
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str, n_ctx: int | None = None) -> str:
    ctx = int(n_ctx) if n_ctx else MODEL_CTX.get(model_name, 32768)
    with _lock:
        if _state["name"] == model_name and _state["model"] is not None and _state.get("ctx") == ctx:
            return f"✓ {model_name} already loaded (ctx {ctx:,})."

        _unload()

        try:
            if model_name in GGUF_MODELS:
                from llama_cpp import Llama

                repo_id = GGUF_MODELS[model_name]
                filename = _best_gguf(repo_id)
                path = hf_hub_download(repo_id=repo_id, filename=filename, token=HF_TOKEN)
                try:
                    _state["model"] = Llama(
                        model_path=path,
                        n_ctx=ctx,
                        n_gpu_layers=-1,
                        verbose=False,
                    )
                except Exception as llama_err:
                    raise RuntimeError(
                        f"llama-cpp-python could not load {filename}.\n"
                        f"This usually means your llama-cpp-python is too old for this "
                        f"model architecture.\n"
                        f"Fix: pip install -U llama-cpp-python\n"
                        f"Original error: {llama_err}"
                    ) from llama_err
                _state["type"] = "gguf"

            elif model_name in HF_MODELS:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from transformers import logging as hf_logging

                repo_id = HF_MODELS[model_name]
                # Suppress the "LOAD REPORT" info message (e.g. original_inv_freq unexpected key)
                prev_level = hf_logging.get_verbosity()
                hf_logging.set_verbosity_error()
                try:
                    _state["tokenizer"] = AutoTokenizer.from_pretrained(repo_id, token=HF_TOKEN)
                    _state["model"] = AutoModelForCausalLM.from_pretrained(
                        repo_id,
                        token=HF_TOKEN,
                        torch_dtype=(
                            torch.bfloat16 if torch.cuda.is_available() else torch.float32
                        ),
                        device_map="auto",
                    )
                finally:
                    hf_logging.set_verbosity(prev_level)
                _state["model"].eval()
                _state["type"] = "hf"

            else:
                return f"✗ Unknown model: {model_name}"

            _state["name"] = model_name
            _state["ctx"] = ctx
            return f"✓ Loaded: {model_name}  (ctx {ctx:,})"

        except Exception as exc:
            _unload()
            return f"✗ Failed to load {model_name}:\n{exc}"


def on_model_select(model_name: str) -> tuple[int, str]:
    """Called when the dropdown changes: set default ctx and load the model."""
    ctx = MODEL_CTX.get(model_name, 32768)
    status = load_model(model_name, ctx)
    return ctx, status


# ── Inference ─────────────────────────────────────────────────────────────────

def generate(
    model_name: str,
    context: str,
    query: str,
    temperature: float,
    max_tokens: int,
):
    """Yields (response_text, stats_text) tuples."""

    if not query.strip():
        yield "⚠ Please enter a query.", ""
        return

    if _state["name"] != model_name or _state["model"] is None:
        yield f"⏳ Loading {model_name}…", ""
        status = load_model(model_name)
        if status.startswith("✗"):
            yield status, ""
            return

    try:
        # ── GGUF via llama-cpp ────────────────────────────────────────────────
        if _state["type"] == "gguf":
            llm = _state["model"]
            messages = []
            if context.strip():
                messages.append({"role": "system", "content": context.strip()})
            messages.append({"role": "user", "content": query.strip()})

            # Approximate prompt token count (raw text, pre-template overhead)
            raw_prompt = (context.strip() + "\n" + query.strip()).strip()
            prompt_tokens = len(llm.tokenize(raw_prompt.encode("utf-8")))

            t_start = time.perf_counter()
            t_first: float | None = None
            gen_count = 0
            response = ""

            for chunk in llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            ):
                delta = chunk["choices"][0]["delta"].get("content", "")
                if not delta:
                    continue

                now = time.perf_counter()

                if t_first is None:
                    t_first = now
                    prefill_s = t_first - t_start
                    stats = _fmt_stats(prompt_tokens, prefill_s, 0, 0.0)
                else:
                    gen_count += 1  # each streaming chunk = 1 decoded token
                    gen_s = now - t_first
                    stats = _fmt_stats(prompt_tokens, prefill_s, gen_count, gen_s)

                response += delta
                yield response, stats

        # ── TorchAO via transformers ──────────────────────────────────────────
        elif _state["type"] == "hf":
            import torch
            from transformers import TextIteratorStreamer

            tok = _state["tokenizer"]
            mod = _state["model"]
            device = next(mod.parameters()).device

            messages = []
            if context.strip():
                messages.append({"role": "system", "content": context.strip()})
            messages.append({"role": "user", "content": query.strip()})

            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tok(text, return_tensors="pt").to(device)
            prompt_tokens = inputs["input_ids"].shape[1]

            streamer = TextIteratorStreamer(
                tok, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                streamer=streamer,
            )

            t_start = time.perf_counter()
            t_first: float | None = None
            gen_count = 0
            prefill_s = 0.0
            response = ""

            thread = threading.Thread(target=mod.generate, kwargs=gen_kwargs)
            thread.start()

            for new_text in streamer:
                if not new_text:
                    continue
                now = time.perf_counter()

                if t_first is None:
                    t_first = now
                    prefill_s = t_first - t_start

                # Approximate token count for streamed text chunks
                chunk_toks = len(tok.encode(new_text))
                gen_count += chunk_toks
                gen_s = now - t_first if t_first else 0.0
                response += new_text
                yield response, _fmt_stats(prompt_tokens, prefill_s, gen_count, gen_s)

            thread.join()

    except Exception as exc:
        yield f"✗ Generation error:\n{exc}", ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EMPTY_STATS = "Prompt: — tok | — tok/s  ·  Generation: — tok | — tok/s"

CUSTOM_CSS = """
.gradio-container { max-width: 1120px !important; }

#app-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 0; }
#app-sub   { color: #64748b; font-size: 0.85rem; margin-top: 2px; }

.section-label {
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 0.72rem;
    font-weight: 700;
    color: #94a3b8;
    margin: 14px 0 4px 0;
}

#stats {
    color: #60a5fa;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 0.78rem;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 8px 10px;
}

#response textarea { min-height: 260px; }
"""

THEME = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="KCC Model Chat") as demo:
    gr.Markdown("KCC Model Chat", elem_id="app-title")
    gr.Markdown("Fine-tuned agricultural advisory models · KCC dataset", elem_id="app-sub")

    with gr.Row(equal_height=False):

        # ── Left pane: Chat ───────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("Query", elem_classes="section-label")
            query_box = gr.Textbox(
                placeholder="Ask a question about farming, crops, pest management…",
                lines=4,
                show_label=False,
            )
            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary", scale=2)
                clear_btn  = gr.Button("Clear", scale=1)

            gr.Markdown("Response", elem_classes="section-label")
            output_box = gr.Textbox(
                show_label=False,
                interactive=False,
                lines=12,
                elem_id="response",
            )
            stats_box = gr.Markdown(EMPTY_STATS, elem_id="stats")

        # ── Right pane: Settings ──────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("Model", elem_classes="section-label")
            model_dd = gr.Dropdown(
                choices=ALL_MODELS,
                value="Llama 3.2 1B (GGUF)",
                show_label=False,
            )
            ctx_box = gr.Number(
                value=MODEL_CTX["Llama 3.2 1B (GGUF)"],
                label="Context length (tokens)",
                precision=0,
                minimum=512,
            )
            load_status = gr.Textbox(
                value="— not loaded",
                show_label=False,
                interactive=False,
            )

            gr.Markdown("Parameters", elem_classes="section-label")
            temperature = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                label="Temperature",
            )
            max_tokens = gr.Slider(
                minimum=64, maximum=2048, value=512, step=64,
                label="Max tokens",
            )

            gr.Markdown("System Context", elem_classes="section-label")
            context_box = gr.Textbox(
                placeholder="E.g. You are a helpful KCC agricultural advisor.",
                lines=4,
                show_label=False,
            )

    # ── Event wiring ──────────────────────────────────────────────────────────
    demo.load(
        fn=on_model_select,
        inputs=[model_dd],
        outputs=[ctx_box, load_status],
    )
    model_dd.change(
        fn=on_model_select,
        inputs=[model_dd],
        outputs=[ctx_box, load_status],
    )

    submit_btn.click(
        fn=generate,
        inputs=[model_dd, context_box, query_box, temperature, max_tokens],
        outputs=[output_box, stats_box],
    )
    clear_btn.click(
        fn=lambda: ("", EMPTY_STATS),
        outputs=[output_box, stats_box],
    )
    query_box.submit(
        fn=generate,
        inputs=[model_dd, context_box, query_box, temperature, max_tokens],
        outputs=[output_box, stats_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
