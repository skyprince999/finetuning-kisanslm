# Fine-tuned Gemma 3n → GGUF: A Field Guide

*Converting a LoRA-fine-tuned Gemma 3n into a quantized GGUF that runs on laptops, phones, Ollama, and llama.cpp — and why it keeps breaking.*

**The one-sentence summary:** Most failures here are **one CUDA version mismatch wearing several disguises**, layered on top of **a new model architecture that the conversion tooling only partially understands.** Debug by collapsing many symptoms back to those two real causes.

---

## The pipeline at a glance

| # | Stage | ELI5 | Technical | Where it fails |
|---|-------|------|-----------|----------------|
| 1 | **Environment setup** | Order your tools | Install torch, transformers, peft, etc. | ⚠️ **The big one.** Latest `torch` pulls a **CUDA 13.0** build, but Colab only has CUDA **12.x** runtime libs. Every compiled package mismatches. |
| 2 | **Load base model** | Bring out the plain robot | Load `gemma-3n-E2B-it` in bf16 on CPU | Needs recent transformers + HF auth (model is gated) |
| 3 | **Apply LoRA adapter** | Clip on your custom chip | PEFT loads the adapter onto the base | ⚠️ Masked `BloomPreTrainedModel` import error; ⚠️ bitsandbytes load error (non-fatal noise) |
| 4 | **Merge weights** | Glue the chip on permanently | `merge_and_unload()` folds `B·A·scale` into base weights | ⚠️ Can produce **empty shards silently** — no exception thrown |
| 5 | **Save model + tokenizer** | Box it up | `save_pretrained()` | ⚠️ Gemma 3n's "tokenizer" is a multimodal **processor** — wrong call signature misfires |
| 6 | **Verify the merge** | Check the box isn't empty | Sum shard sizes, assert none are 0 bytes | Skipping this = discovering failure an hour later |
| 7 | **Convert to GGUF** | Translate to laptop language | `convert_hf_to_gguf.py` → BF16 GGUF | ⚠️ **Multimodal wall.** Only the **text** path converts; audio/vision towers are dropped (expected) |
| 8 | **Quantize** | Shrink it to fit | `llama-quantize` → Q4_K_M etc. | ⚠️ `per_layer_token_embd` must stay Q8_0 or it **breaks in Ollama** |
| 9 | **Save / upload** | Hand out copies | Copy to Drive, push to HF | Watch sync size mismatches; never hardcode write tokens |
| 10 | **Run it** | Switch the robot on | llama.cpp / Ollama inference | ⚠️ Runtime auto-adds `<bos>` — a second one corrupts prompts |

---

## One root cause, three masks

A single **CUDA ABI mismatch** (torch built for CUDA 13.0, host has CUDA 12.x) surfaces as three unrelated-looking errors. `pip` never catches it because it's a binary-load failure, not a Python version conflict.

| What you see | What it actually is |
|--------------|---------------------|
| `PyTorch CUDA 13.0 vs TorchAudio CUDA 12.8` (RuntimeError) | The mismatch, stated plainly |
| `libnvJitLink.so.13: cannot open shared object file` (bitsandbytes) | The loader wants the CUDA-13 lib; only `.so.12` exists. **Non-fatal** — execution continues |
| `Could not import module 'BloomPreTrainedModel'` (peft) | transformers' `_LazyModule` **catches the real CUDA error and re-raises a generic message**. Not a peft problem at all |

**Debugging move:** read past the top-level message to the `from e` cause underneath, or reproduce directly with `import torchaudio`.

---

## Errors that lie — in both directions

- **Hide the real problem:** the peft `BloomPreTrainedModel` message disguises a CUDA loader failure as a missing model class.
- **Look fatal but aren't:** bitsandbytes logs at `ERROR` level, but a bf16 CPU merge never calls its GPU kernels, so execution sails right past. The log is inert.

Knowing which is which is the actual skill.

---

## Collateral dependencies (never needed, pure failure surface)

`torchaudio`, `torchvision`, `torchao`, and `bitsandbytes` are **not required** for a bf16 CPU merge → GGUF flow. They're dragged in by the install and become things that can only break. For this notebook, removing them is more robust than version-matching them.

```bash
pip uninstall -y torchaudio torchao bitsandbytes
```

---

## Quantization quick reference

The name encodes a **bit budget** (Q4/Q5/Q6/Q8) and a **scheme**:
- `_0` / `_1` = legacy, simpler, slightly worse per bit
- `_K` = k-quants (super-blocks + per-block scales), the modern default
- `_S` / `_M` / `_L` = mixed precision; important layers bumped higher (so `Q4_K_M` ≠ uniformly 4-bit)

| Quant | ~bits/wt | ~size (5.4B) | Quality | Pick when |
|-------|----------|--------------|---------|-----------|
| **Q4_K_M** | ~4.8 | ~3.5 GB | Good, minor loss | **Default sweet spot** |
| Q5_K_M | ~5.7 | ~4.1 GB | Very good | Want a margin over Q4 |
| Q6_K | ~6.6 | ~4.7 GB | Near-lossless | Quality matters, size barely does |
| Q8_0 | ~8.5 | ~5.9 GB | Effectively lossless | Archival / base for other quants |
| BF16 | 16 | ~11 GB | The original | Master copy, not for running |

**Rule of thumb:** start at Q4_K_M; only go higher if you can *perceive* the difference. Quality falls off gently from Q8→Q4, then off a cliff below Q4. Q4_K_M sits at the knee of that curve.

---

## The Gemma 3n asterisks (what makes this model special)

Gemma 3n is not a vanilla decoder. It uses a **MatFormer (nested/elastic) architecture**, **per-layer token embeddings**, and separate **audio + vision encoders**. That novelty is the source of the model-specific failures:

1. **Multimodal wall** — GGUF/llama.cpp support only the text path, so conversion drops the encoders. This is why Unsloth's one-shot GGUF export fails for Gemma 3n and forces the manual merge → llama.cpp route. *Expected, not corruption.*
2. **Embedding precision** — `per_layer_token_embd` must be Q8_0 (or Q4_0/1, Q5_0/1). A blanket Q4_K_M under-quantizes it: **runs in llama.cpp, errors in Ollama.** Override at quantize time:
   ```bash
   llama-quantize --token-embedding-type q8_0 model.BF16.gguf model.Q4_K_M.gguf Q4_K_M
   ```
3. **Metadata bug (historical)** — `add_shared_kv_layers` was emitted as float32 where Ollama expected uint32.
4. **`<bos>` doubling** — Gemma's chat template prepends `<bos>`; llama.cpp/Ollama auto-prepend it too. Strip one. (Training-side: `.removeprefix("<bos>")`.)

**"Converted successfully" ≠ "works in Ollama."** Test in your actual target runtime.

---

## Fixes cheat sheet

| Symptom | Fix |
|---------|-----|
| Any CUDA `.so.13` / version-mismatch error | Don't upgrade torch on Colab; install only `transformers peft accelerate` on top of the bundled stack |
| `Could not import module 'BloomPreTrainedModel'` | Same as above — it's the CUDA mismatch in disguise |
| bitsandbytes load error during CPU merge | Ignore it (non-fatal), or `pip uninstall -y bitsandbytes` |
| Empty `.safetensors` after merge | Verify shard sizes before converting; re-run merge |
| GGUF works in llama.cpp, fails in Ollama | Force `--token-embedding-type q8_0` at quantize |
| Garbled output | Remove the duplicate `<bos>` |

---

## The takeaway

Two real causes underlie almost everything:
1. **A bleeding-edge CUDA stack** on a managed environment → ABI mismatches refracted through each library's error style.
2. **A novel architecture** the conversion/quantization tooling only partially models.

Stop chasing individual messages. Collapse them back to these two, and the saga gets short.
