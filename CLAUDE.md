# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A ComfyUI custom-node pack ("Painter Nodes") for image/video generation workflows: Wan2.2 image-to-video, LTXV sampling, audio-driven video (InfiniteTalk / Humo / S2V), Flux & Qwen image editing, video combine/upscale, VRAM management, and prompt utilities. Distributed as a folder dropped into `ComfyUI/custom_nodes/`.

There is no build step, no test suite, and no linter configured. Runtime dependencies are listed in `requirements.txt` (`soundfile`, `numpy`); everything else (`torch`, `comfy.*`, `node_helpers`, `folder_paths`, `latent_preview`, `comfy_api.latest`) is provided by the host ComfyUI process at import time. The code cannot be exercised standalone — to test a change, restart ComfyUI and load the node in a workflow.

## Architecture

### Node registration

`__init__.py` is the entry point ComfyUI loads. It iterates an explicit `NODE_MODULES` list, importing each module via `importlib` inside a `try/except` so a single broken node doesn't take down the whole pack — failures are collected and printed at startup. Each module is expected to expose two dicts: `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`, which are merged into the package-level dicts that ComfyUI consumes.

**When adding a new node module, you must append it to `NODE_MODULES` in `__init__.py`.** Files not listed there are not loaded (e.g. `PainterAI2V_fixed.py` is present in the tree but not registered — treat such files as scratch/dev unless you intend to wire them up).

`WEB_DIRECTORY = "./web/js"` exposes browser-side extensions. Only the three nodes with custom UI behavior have JS files there (`PainterFluxImageEdit`, `PainterQwenImageEditPlus`, `PainterVideoCombine`); all other nodes are pure Python.

### Two coexisting node APIs

The pack mixes two ComfyUI node-definition styles, and you should match whichever the file you're editing already uses:

- **Legacy class API** (most files, e.g. `PainterSampler.py`, `PainterVRAM.py`, `PainterVideoCombine.py`): classmethod `INPUT_TYPES`, class attributes `RETURN_TYPES` / `RETURN_NAMES` / `FUNCTION` / `CATEGORY` / `OUTPUT_NODE`, plus an instance method named by `FUNCTION`.
- **New schema API** (`PainterI2V.py`, `PainterI2VAdvanced.py`, `PainterAI2V.py`, `PainterAV2V.py`, `PainterFLF2V.py`, `PainterHumoAI2V.py`, `PainterS2Vplus.py`, `PainterSamplerLTXV.py`): subclass `io.ComfyNode` from `comfy_api.latest`, define `define_schema()` returning `io.Schema`, implement `execute()` returning `io.NodeOutput`, and pair with a `ComfyExtension` subclass plus an `async def comfy_entrypoint()`.

Both styles still register through the same `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` dicts at module bottom — that's the contract `__init__.py` relies on.

### Recurring conditioning patterns

Image-to-video and frame-to-video nodes follow a consistent shape that's worth recognizing before editing them:

1. Allocate a zero latent of shape `[B, 16, ((length-1)//4)+1, H//8, W//8]` on `comfy.model_management.intermediate_device()`.
2. Build a per-frame image tensor (real frames at known positions, 0.5-gray elsewhere), VAE-encode it, and a matching `concat_mask` (0 where the frame is constrained, 1 elsewhere).
3. Inject `concat_latent_image` and `concat_mask` into both positive and negative conditioning via `node_helpers.conditioning_set_values(...)`.
4. Optionally add `reference_latents` (append=True) and `clip_vision_output`.

`PainterI2V.py` additionally implements a "motion amplitude" trick — center-then-scale the latent diff between the start frame and gray frames — specifically to counteract the slow-motion artifact of 4-step LoRAs (lightx2v). Don't remove the `torch.clamp(scaled_latent, -6, 6)`; the README and code comments treat this whole pipeline as load-bearing for that fix.

### Sampler structure

`PainterSampler` and `PainterSamplerLTXV` implement two-phase ("high-noise then low-noise") sampling with separate `high_model` / `low_model` and `high_cfg` / `low_cfg`, switching at `switch_at_step`. Phase 1 runs with noise enabled and `force_full_denoise=False`; phase 2 runs with `disable_noise=True` on the phase-1 output. Both files keep a private copy of the upstream `common_ksampler` rather than importing it.

### VRAM control

`PainterVRAM.py` mutates `comfy.model_management.EXTRA_RESERVED_VRAM` directly. In `auto` mode it reads current GPU usage via `pynvml` (optional dep — if missing, auto silently falls back to manual). It uses an `AlwaysEqualProxy("*")` wildcard type so the node accepts any input/output type — this is intentional, the node is meant to be inserted as a passthrough barrier in arbitrary workflows. When `anything` is not connected, it returns an `ExecutionBlocker` to stop downstream execution.

### Video output

`PainterVideoCombine.py` shells out to ffmpeg (via `imageio_ffmpeg.get_ffmpeg_exe()`, falling back to `"ffmpeg"` on PATH). It writes raw RGB frames to a temp file first and pipes that into ffmpeg via stdin redirection — this is a deliberate workaround for pipe deadlocks on large frame counts, don't "simplify" it back to a direct `Popen(stdin=PIPE)` write loop. Audio, when supplied, is written to a temp WAV and added as a second `-i` input.

## Conventions specific to this repo

- Node display names in `NODE_DISPLAY_NAME_MAPPINGS` are user-facing and often contain parenthetical hints (e.g. `"PainterI2V (Wan2.2 Slow-Motion Fix)"`). Match this style for new nodes.
- Categories are inconsistent across the pack (`Painter/Prompt`, `Painter/Video`, `sampling/painter`, `conditioning/video_models`, `advanced/conditioning`, `VRAM`). When adding a node, pick whichever category an analogous existing node uses rather than inventing a new namespace.
- Comments and log strings mix Chinese and English; the README is primarily Chinese. Don't translate existing comments unless asked.
- Workflows under `workflows/` are reference `.json` files users load in ComfyUI — they are not test fixtures and are not consumed by any code in this repo.
