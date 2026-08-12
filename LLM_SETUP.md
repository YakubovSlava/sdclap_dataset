# LLM setup and configuration (stage 1.2 — speech-manner extraction)

This document specifies the exact local LLM setup used by `scripts/03_extract_speech_fragments.py`
to extract direct-speech fragments and their manner-of-speech descriptions from book text
(stage 1.2 of the pipeline). It is a condensed, reproducibility-focused summary of the fuller
session log kept in the main repository at `docs/llm_setup.md`.

## Model

- **Checkpoint**: [`unsloth/gemma-4-E4B-it-qat-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF),
  quantization `UD-Q4_K_XL` (~4.2 GB), file `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`.
- **MTP draft model** (speculative decoding): `mtp-gemma-4-E4B-it.gguf` (57 MB), same repo.
- The multimodal projector (`mmproj-BF16.gguf`) shipped with the checkpoint is **not** needed for
  text-only extraction and was not used.

## Serving

Served locally via **llama.cpp** (`llama-server`), built from source with CUDA support, exposing
an OpenAI-compatible `/v1/chat/completions` endpoint. Structured output enforced via
`response_format: json_schema` (grammar-constrained decoding) from the client side
(`03_extract_speech_fragments.py`), not shown here.

Exact launch command used for the extraction run that produced the published dataset:

```sh
./llama.cpp/build/bin/llama-server \
  -m gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  -md gemma-4-E4B-it-qat-GGUF/mtp-gemma-4-E4B-it.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -a gemma-4-e4b \
  -ngl 999 -fa off \
  -c $((16 * 8192)) \
  -np 16 \
  -ub 512 -b 1024 \
  --host 127.0.0.1 --port 1234
```

| Flag | Value | Why |
|---|---|---|
| `-m` | main weights | UD-Q4_K_XL, 4.2 GB, fits comfortably in 12 GB VRAM |
| `-md` / `--spec-type draft-mtp` | MTP draft model | speculative decoding, **+27% throughput** at no quality cost (measured on real extraction workload: 685.5s/130 chunks without MTP vs 499.8s/130 chunks with MTP, draft acceptance 71.6% on an isolated request) |
| `-a gemma-4-e4b` | API model alias | name reported in the `"model"` field of requests |
| `-ngl 999` | offload all layers to GPU | weights are small enough for full offload |
| `-fa off` | flash attention disabled | **required for MTP** — speculative decoding does not work with `-fa on` in this llama.cpp version |
| `-c $((16*8192))` | total context = 131072, i.e. 8192 tokens/slot | with `-fa off` the KV-cache is heavier; 4096 tokens/slot (the value used at `-np 32`) was too small and caused JSON truncation on multi-fragment chunks |
| `-np 16` | 16 concurrent slots | must match `--workers` in `03_extract_speech_fragments.py` |
| `-ub 512 -b 1024` | physical/logical batch size (deliberately modest, not 2048/8192) | with `-fa off` + MTP, larger batches caused `cudaMalloc failed: out of memory` |
| `--host 127.0.0.1 --port 1234` | local server | no external exposure |

`--cache-reuse`/`-kvu` (unified KV cache) are **not** used — tested and found not to help with
this model/llama.cpp version (see `docs/llm_setup.md` in the main repository for the full
investigation).

Health check: `curl http://127.0.0.1:1234/health`.

## Key generation-config decisions

- **Chain-of-thought ("thinking") left unbounded** (no `--reasoning-budget`, no
  `chat_template_kwargs.enable_thinking: false`). Both a fully-disabled and a budget-limited
  (`200` tokens) thinking mode were tested and rejected: they made the model violate the
  extraction prompt significantly more often, inserting placeholder descriptions (`"сказал"`,
  `"—"`, empty string) instead of correctly omitting a fragment with no valid manner description.
- **Prompt explicitly restricts `description` to what is audible**, not visible — the model is
  instructed to only describe properties a listener could infer from audio alone, with an
  explicit exclusion list for gestures/visual detail (bowed, smiled, frowned, nodded, shrugged,
  etc.), even when adjacent to the quoted line.
- **Client-side post-filter** (`is_junk_description()` in `03_extract_speech_fragments.py`) drops
  three categories of residual junk the model still occasionally produces despite the prompt: (1)
  explicit placeholders (empty, `—`, `null`, "no manner description", bare "no"), (2) bare neutral
  speech verbs with no coloring (`сказал`/`ответил`/`спросил`/... optionally + a short pronoun),
  (3) cases where the model copies the `speech` text itself into `description` instead of
  describing how it was said.
- A two-stage triage (cheap yes/no pre-check before the full JSON extraction call) was tried as a
  cost-saving measure and **abandoned**: with unbounded thinking, the triage call still pays the
  full prefill cost of the chunk and only saves on the (comparatively small) final-JSON generation
  cost, making it a net slowdown, not a saving.

## Structured output

Extraction responses are constrained to a JSON schema via llama.cpp's grammar-constrained
decoding (`response_format: json_schema` in the request), not free-form generation — this,
together with the post-filter above, is what keeps the extracted `speech`/`description` pairs in
a consistent, parseable shape across the full corpus.

## Exact prompt and JSON schema

The full system prompt (`SYSTEM_PROMPT`) and the structured-output schema (`FRAGMENT_SCHEMA`) used
for extraction are not duplicated here — they are the literal, unmodified source in
[`scripts/03_extract_speech_fragments.py`](scripts/03_extract_speech_fragments.py) (`SYSTEM_PROMPT`
at line 90, `FRAGMENT_SCHEMA` at line 58), included verbatim in this package for exact
reproducibility. Both are passed as-is to the `/v1/chat/completions` request (`call_llm()` in the
same file) together with the launch config above.

For the full narrative of how this configuration was arrived at (including configurations that
were tried and rejected), see `docs/llm_setup.md` and the corresponding `history.md` entries
(2026-07-11/2026-07-12) in the main repository.
