# Wan2GP / WanGP — Repository Analysis

Analysis of `deepbeepmeep/Wan2GP` as mirrored into `klingesh/Higgsfield`.

- **Version analysed:** WanGP v12.52 (`wgp.py:150`)
- **Commit:** `7f06022de4` (`main`)
- **Upstream:** 8,576 stars · 1,330 forks · created 2025-02-27
- **Scale:** 525,325 LOC Python across 1,653 `.py` files · 2,324 files total · 87 MB history · 1,660 commits

---

## 1. What this project is

A **Gradio desktop web-app for running large generative video/image/audio models on consumer GPUs**. The tagline "for the GPU Poor" is the entire thesis: upstream research models (Wan, LTX-2, Hunyuan, Flux, Qwen-Image) normally need 24–80 GB VRAM; WanGP runs them on 10–12 GB by streaming weights layer-by-layer between pinned system RAM and VRAM.

The engineering value is **not** the models — those are vendored from upstream research repos. It is the productisation layer: memory management, a declarative model registry, LoRA scheduling, sliding-window long-video generation, queueing, and reproducible outputs.

**Author dominance:** `DeepBeepMeep` accounts for 1,238 of ~1,620 commits (two capitalisations of the same identity). Second contributor `Chris Malone` has 306. Development is sustained and current — 41 commits in Aug 2026, 179 in Feb 2026 — but this is effectively a single-maintainer project.

---

## 2. Architecture at a glance

```
wgp.py  (13,695 lines — CLI + registry + Gradio UI + queue + orchestrator)
  │
  ├── defaults/*.json (215 files)  ← declarative model definitions
  ├── finetunes/*.json             ← user-defined models, same schema
  │
  ├── models/<family>/…_handler.py (26 handlers)  ← duck-typed per-family adapter
  │     └── models/<family>/…  vendored upstream model code (~1,059 files)
  │
  ├── shared/          cross-cutting: attention, download, plugins, deepy agent, gradio widgets
  ├── preprocessing/   control-signal extractors (pose, depth, canny, SAM-3, matting, vocals)
  ├── postprocessing/  upscalers (FlashVSR, SeedVR2), RIFE interpolation, audio (MMAudio, SeedVC)
  ├── plugins/         5 system + 5 bundled plugins, dynamically imported
  └── profiles/        55 preset/accelerator JSONs (step-distill LoRA presets)
```

### The monolith
`wgp.py` is simultaneously the CLI entry point, config loader, model registry, Gradio UI builder, task queue and generation orchestrator. Critically, **much of its work happens at module import time** — config load, `refresh_model_defs()`, deprecated-file deletion, even eager model loading. Importing `wgp.py` has global side effects, so it cannot be used as a library or unit-tested. The in-app agent (`shared/deepy/engine.py`) reaches back into it via `sys.modules["__main__"]`.

Two god-functions dominate:

| Function | Lines | Role |
|---|---|---|
| `generate_media()` | 6440–8130 (~1,700) | Generation orchestrator, ~200 keyword params |
| `generate_media_tab()` | 11072–13061 (~2,000) | Builds the entire main UI *and* wires every event |

---

## 3. The model registry — the best idea in the codebase

A model type is **the basename of a JSON file**. `refresh_model_defs()` (`wgp.py:3170`) globs `defaults/*.json` + `finetunes/*.json`; the `"model"` key becomes the model definition and **every other key becomes default UI form values**.

Minimal definition (`defaults/t2v.json`):

```json
{"model": {
  "name": "Wan2.1 Text2video 14B",
  "architecture": "t2v",
  "URLs": ["...wan2.1_text2video_14B_mbf16.safetensors",
           "...wan2.1_text2video_14B_quanto_mbf16_int8.safetensors",
           "...wan2.1_text2video_14B_quanto_mfp16_int8.safetensors"]
}}
```

`URLs` is an **ordered quantisation-variant list** — `get_model_filename()` (`wgp.py:2880`) picks by matching `int8`/`fp8`/`bf16`/`fp16` tokens in the filename. `URLs` can also be a *string naming another model type*, resolved recursively by `get_model_recursive_prop()` (`wgp.py:2849`). This gives genuine composition — Vace 14B is *t2v weights + a Vace ControlNet module*:

```json
{"model": {"name": "Vace 14B", "architecture": "vace_14B",
  "modules": [["...wan2.1_Vace_14B_module_mbf16.safetensors", "…int8…"]],
  "URLs": "t2v"}}
```

**Consequence:** adding a *variant* of a supported architecture is a single JSON file, zero code. That is why 215 model definitions sit on top of only 26 handlers. Family breakdown by architecture: ltx2 (23), hunyuan (16), vace (14), flux (11), z-image (9), kandinsky5 (9), i2v (9), ace-step (9), t2v (8), qwen (8), ovi (6), flux2 (6), plus minimax/magi/lucy/longcat/krea2 and 9 TTS families.

### Handlers are duck-typed, with no contract
`family_handlers` (`wgp.py:2424`) is a hard-coded list of 26 module paths. Each module exposes `class family_handler:` with all-static methods discovered by `getattr` — **there is no ABC or Protocol**. The surface (see `models/wan/wan_handler.py`):

`query_supported_types` · `query_model_def` · `query_model_files` · `load_model` → `(model, pipe)` · `fix_settings` · `update_default_settings` · `validate_generative_settings`

Worse, **capabilities are inferred from string tests on architecture names** — `test_vace()`, `test_class_i2v()`, `test_multitalk()`, `test_wan_5B()` (`wan_handler.py:29–69`). Model-specific special-cases then leak upward into `generate_media` (e.g. `model_def.get("joyai_echo", False)` at `wgp.py:7576`).

The one hard contract is `wan_model.generate(**~120 kwargs)`, dispatched from a single 130-line keyword blob at `wgp.py:7593–7727`. Every family must absorb the full surface via `**kwargs`, so a newly-added or mistyped kwarg **silently no-ops** in most handlers. `models/ideogram4/` is a remote-API model implementing the same interface — proof the abstraction is at least general.

---

## 4. The "GPU Poor" machinery

### mmgp is the whole strategy
`from mmgp import offload, safetensors2, profile_type, quant_router` (`wgp.py:38`). mmgp ("Memory Management for the GPU Poor") is a separate pip package by the same author, **hard-pinned to `mmgp==3.7.12`** — `wgp.py:169-174` reads the installed version and **calls `exit()` on mismatch**. Handlers merely return a dict of named submodules; `offload.profile(...)` (`wgp.py:4034`) does all tensor movement. WanGP itself never moves a tensor.

### Memory profiles (`wgp.py:10908`)

| # | mmgp name | Requirement | Trade-off |
|---|---|---|---|
| 1 | HighRAM_HighVRAM | 64 GB RAM / 24 GB VRAM | Fastest for short videos |
| 2 | HighRAM_LowVRAM | 64 GB RAM / 12 GB VRAM | Most versatile; streamed from pinned RAM |
| 3 | LowRAM_HighVRAM | 32 GB RAM / 24 GB VRAM | All in VRAM, `budgets={"*":"70%"}` |
| 3.5 | VeryLowRAM_HighVRAM | — | As 3, no pinned memory → less RAM |
| **4** | **LowRAM_LowVRAM** | **32 GB RAM / 12 GB VRAM** | **Default.** Layer-by-layer streaming |
| 4.5 | Profile 4+ | — | As 4, async transfers off → slower, less VRAM |
| 5 | VeryLowRAM_LowVRAM | 24 GB RAM / 10 GB VRAM | Fail-safe, slowest |

`init_pipe()` (`wgp.py:3796`) translates a profile into mmgp settings. Profiles 2/4/5 get per-module VRAM budgets — transformer and text_encoder capped at 100 MB, everything else `max(1000 if profile==5 else 3000, preload)` MB. Profiles 3/4 pin `transformer`/`transformer2` for two-expert Wan 2.2 models. Separate defaults exist per output type (`video_profile`, `image_profile`, `audio_profile`).

### Quantisation: pre-quantised downloads, not runtime conversion
The normal path is **downloading an already-quantised checkpoint** selected by filename token. On-the-fly quantisation is opt-in per model and narrowly gated (`wgp.py:3936`): it requires `auto_quantize: true` in the def, and is disabled when `modules` are declared or the filename already contains `quanto`. Custom quant formats are registered into mmgp's router at import (`wgp.py:183`): scaled_fp8, nvfp4, bnb_nf4, nunchaku int4/fp4, asym_w4a8_int8, int8_convrot, gguf. `--save-quantized` produces new quanto files and **rewrites the finetune JSON in place** (`wgp.py:3450`).

### LoRAs
Applied *on top of* the offloaded/quantised model, never merged into the checkpoint (except "accelerator profiles", which request merge-before). Per-step and per-phase multiplier schedules are parsed by `parse_loras_multipliers()`, then `offload.load_loras_into_model(...)` (`wgp.py:6786`). Note `pinnedLora = not is_mps and loaded_profile != 5` — profile 5 refuses to pin LoRA weights to save RAM.

### `profiles/` is *not* memory profiles
Despite the name, the 55 files in `profiles/` are the **preset/accelerator library** — settings JSONs that mostly wire step-distillation LoRAs (`profiles/wan/Lightx2v t2v Cfg Step Distill Rank32 - 4 Steps.json`, `profiles/flux/Turbo Alpha 10 Steps.json`). An easy and consequential source of confusion.

### Attention backends (`shared/attention.py`)
Every backend is imported in `try/except ImportError`, so a missing package silently disables a mode: sdpa, flash-attn 2, flash-attn 3, xformers, sage, sage2, sage3, radial (block-sparse), sol. Auto-selection order is **sage2 → sage → sdpa** (`:278`). Capability filtering removes modes the GPU can't run (sage3 needs compute capability ≥ 10; MPS gets only `["sdpa","auto"]`).

**Important caveat:** selecting "sage2" is not a guarantee. There are runtime fallbacks — with an attention mask, sage2 degrades to sdpa with a one-time warning (`:383-395`); "radial" collapses to sage2 (`:421`); pre-Ada GPUs downgrade the whole sage family (`:299`).

---

## 5. Generation flow

```
[Generate button]
  init_generate                      wgp.py:9404
  → validate_wizard_prompt           wgp.py:9078
  → save_inputs                      wgp.py:10087
  → process_prompt_and_add_tasks     wgp.py:447    expand + validate + enqueue
  → process_tasks  (generator)       wgp.py:8194   THE PUMP
       └── worker thread → generate_media(...)     wgp.py:6440
                             ├── load_models       wgp.py:3909  → offload.profile
                             ├── LoRA resolution   wgp.py:6751
                             ├── sliding-window arithmetic
                             ├── wan_model.generate(**120 kwargs)   wgp.py:7593
                             │      └── models/wan/any2video.py:414  denoising loop
                             │            └── callback → build_callback  wgp.py:4116
                             └── upsample → save → record_file_metadata
  → finalize_generation              wgp.py:4447
```

**Concurrency model:** a worker thread plus an `AsyncStream` command bus (`shared/utils/thread_utils.py`). The worker pushes `status`/`progress`/`preview`/`output`/`error` commands; the Gradio generator drains them and yields trigger timestamps that fan out to gallery/preview refreshes. Pause, abort and early-stop are all funnelled through `build_callback`, which is handed into the denoising loop.

**No database.** State is a Gradio `gr.State` dict; `state["gen"]` holds the queue, progress and file lists, guarded by ad-hoc module-level `lock`/`gen_lock`. Cross-process GPU arbitration is bolted on via `shared/utils/process_locks.py`.

### Two genuinely clever touches

1. **Settings embedded in outputs.** `prepare_inputs_dict()` + `record_file_metadata()` write the full generation settings into the output file's metadata, so dragging a generated video back into the UI reconstitutes the exact recipe. `models_eqv_map`/`models_comp_map` govern which models can accept another's settings.
2. **Sliding-window generation as a first-class concept** — overlap frames, overlap noise, colour correction, discard-last-frames, FlashVSR continuation cache, audio window stitching. This is what lets short-context models produce long videos, and it lives in the orchestrator rather than being duplicated per family.

**Crash resilience:** the queue serialises to a zip (JSON manifest + media attachments), autosaves on error, and auto-resumes on page load.

---

## 6. Extension systems

### Plugins (`shared/utils/plugins.py`, 1,725 lines)
Subclass `WAN2GPPlugin` in `<plugin_dir>/plugin.py`. Hooks cover UI injection (`setup_ui`, `add_tab`, `insert_after`), lifecycle (`on_model_change`, `on_tab_select`), data interception (`register_data_hook`), raw JS injection (`add_custom_js`), and host access (`request_global`/`set_global`).

- **System plugins** (always loaded): `video_mask_creator`, `guides`, `configuration`, `plugin_manager`, `about`
- **Bundled**: `downloads`, `media_flow`, `models_manager` (3,236 lines), `motion_designer`, `sample`
- `plugins.json` at the repo root is a **remote catalogue** of third-party plugins, refreshed from a GitHub raw URL — not the installed set.

### Pre/post-processing registries
Small documented handler contracts, each receiving `init_pipe` and `profile` so they build their own mmgp offload profile and register it for later unloading:
- **Spatial upsamplers** (`postprocessing/spatial_upsamplers.py:78`): lanczos, FlashVSR, SeedVR2, PID, chain-of-zoom, ltx2, WanVaeUpsampler
- **Temporal**: RIFE interpolation
- **Audio**: custom soundtrack, MMAudio, PrismAudio, SeedVC voice conversion, background removal
- **Preprocessing**: DWPose, Depth-Anything v2/v3, canny, scribble, flow, SAM-3 segmentation, MatAnyone matting, vocal extraction

### Other entry points
Beyond the Gradio UI: `--mcp` (MCP server, `wgp.py:13436`), `--ask-deepy` (LLM agent CLI), `--process` (headless queue), plus `shared/api.py` / `api_cli.py` / `api_webui.py`. A Docker path exists (`Dockerfile` on CUDA 12.8.1, parameterised by compute capability).

### Deepy — an in-app LLM agent
`shared/deepy/engine.py` (6,248 lines) is an agentic loop that exposes WanGP's own functions as LLM tools. An `@assistant_tool` decorator builds JSON tool schemas from type annotations, and `_get_main_callable` reaches into `wgp.py`'s module globals. It manages sessions, transcripts, interruption/rollback with KV-prefix-mismatch diagnostics, and can enqueue and inspect real generation tasks. It also does documentation search over `docs/`.

---

## 7. Licensing — read this before doing anything commercial

⚠️ **This is not open-source software.** GitHub reports the licence as `NOASSERTION` because `LICENSE.txt` is a bespoke **"WanGP Community License 2.0"**.

**Permitted:** free use including inside a company · modification for your own use · **selling or licensing the outputs you generate** (credit WanGP when you directly sell an output).

**Prohibited without a separate written commercial licence** (§5.1–5.2):
- Selling WanGP itself
- Paid API, paid SaaS, paid hosted or paid managed access
- White-labelling
- Embedding it in a paid product
- OEM or reseller packages

Third-party code, models and weights **keep their own licences** (`docs/third_party_licenses/` ships Apache-2.0, BSD-3-Clause and MIT texts). The vendored research models each carry upstream terms — several Wan/Hunyuan model licences have their own restrictions that this licence explicitly does not override.

**What this means for your mirror:** you now have a public copy under your own account. Redistribution of source is contemplated by the licence, and `LICENSE.txt` travelled with the clone, which is the right outcome for compliance. But your fork is *not* MIT/Apache, and if you ever wanted to host it as a paid service you would need to talk to the author first.

---

## 8. Security posture

The threat model here is "software you run locally on your own machine", and judged on that basis it is unremarkable for the ML ecosystem. But the numbers are worth knowing:

| Finding | Count | Note |
|---|---|---|
| `torch.load()` **without** `weights_only=True` | **110** | Pickle deserialisation of downloaded `.pth`/`.pt`/`.ckpt` — arbitrary code execution if a weight file is malicious. 47 calls *do* set it, so the codebase knows about the flag. |
| `trust_remote_code=True` | 10 | Executes Python from the model repo at load time (e.g. `shared/prompt_enhancer/qwen35_vl.py:174`) |
| `subprocess(..., shell=True)` | 7 | 6 in `setup.py` with interpolated strings; 1 in a TTS module |
| `eval`/`exec` | 7 | None in `wgp.py` or `shared/utils` |
| `pickle.load` | 1 | |

**Plugin system is arbitrary code execution by design.** `install_plugin_from_url()` git-clones a user-supplied URL then runs `pip install -r requirements.txt` and `pip install -e <dir>`. The only validation is `startswith("https://github.com/")` — no signature or pin checking. Even *displaying plugin metadata* imports and instantiates plugin classes.

**Unrestricted global sharing.** `app.initialize_plugins(globals())` (`wgp.py:13250`) hands the entire `wgp` namespace to plugins, and `set_wgp_global()` (`wgp.py:212`) lets any plugin overwrite any global by name. The `restricted_globals` deny-list exists but **is initialised empty**, so the intended protection is inert.

**Model weights are fetched from live URLs** embedded in the 215 JSON definitions plus `huggingface_hub` calls, cached under `ckpts/`. A compromised or typo-squatted host is a code-execution vector given the `torch.load` situation above.

**Practical advice:** run it in a container or a dedicated user account, install plugins only from sources you trust, and treat `defaults/*.json` as trusted input.

---

## 9. Strengths and weaknesses

**Strengths**
- The declarative registry with recursive `URLs`/`modules` inheritance is genuinely elegant — 215 models on 26 handlers, new variants need zero code.
- mmgp offloading is a real, differentiated capability, not a wrapper.
- Reproducibility via settings-in-metadata is better than most commercial tools.
- Sliding-window long-video logic is centralised rather than duplicated per model.
- Fails soft in the right places: missing attention backends disable modes, LM engines downgrade, queues autosave and resume.
- Documentation is substantial — 21 markdown guides in `docs/`.

**Weaknesses**
- **The monolith.** 13,695 lines with import-time side effects; untestable and unusable as a library.
- **Signature-as-schema.** `process_tasks` derives valid task fields from `inspect.signature(generate_media)` (`wgp.py:8295`). The persisted queue format *is* a function's parameter list — renaming a parameter silently drops saved fields.
- **No interface contracts.** Duck-typed handlers, `hasattr` capability probing, and capability inference from architecture-name string matching.
- **Stringly-typed prompt modes.** `image_prompt_type`/`video_prompt_type` are letter-code strings (`"VAG"`) manipulated by ~40 `refresh_*` handlers with no schema — hence the 530-line `validate_settings`.
- **Error handling by string matching** on CUDA exception text to distinguish VRAM from RAM exhaustion — brittle across torch versions.
- **Mutable module globals** (`wan_model`, `offloadobj`, `transformer_type`, `reload_needed`) with ad-hoc locking; effectively single-GPU, single-generation.
- **UI/logic entanglement.** Business logic raises `gr.Error` and returns `gr.update()` objects, so the four headless entry points must tolerate Gradio types.
- **Version fragility.** Hard `exit()` on mmgp version mismatch; `gradio==5.29.0` pinned with three explicit monkey-patch modules; `transformers==4.54.0` pinned. Dependencies include a nightly ONNX Runtime index and author-hosted wheels for insightface/chumpy/smplfitter.
- **No test suite.**

---

## 10. If you want to work on this

- **Add a model variant:** drop a JSON in `finetunes/`. No code. Start from `defaults/t2v.json`; use `"URLs": "<other_model_type>"` to inherit weights.
- **Add an architecture:** write `models/<family>/<family>_handler.py` with a `family_handler` class, add it to `family_handlers` (`wgp.py:2424`). Copy `models/wan/wan_handler.py` as the reference.
- **Tune for low VRAM:** profile 4 is the default; try 5, then `--perc-reserved-mem-max`, then lower resolution. `--preload` pins MB of transformer in VRAM.
- **Read first:** `docs/OVERVIEW.md`, `docs/MODELS.md`, `docs/FINETUNES.md`, `docs/PLUGINS.md`, `docs/TROUBLESHOOTING.md`.
- **Highest-value refactor** if you plan to extend it: define a `Protocol` for `family_handler` and replace the `wan_model.generate(**120 kwargs)` blob with a dataclass. That alone would eliminate the silent-kwarg-drop class of bug.
