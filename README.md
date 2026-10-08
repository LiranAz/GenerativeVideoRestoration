<div align="center">
<h2>Generative Video Restoration</h2>
<p>Pixel-space video super-resolution / restoration with a diffusion transformer, forked from
<a href="https://github.com/kunncheng/DiT-SR">DiT-SR</a> (<a href="https://arxiv.org/abs/2409.19589">arXiv 2409.19589</a>).</p>
</div>

> **Status: research code, untrained.** The whole pipeline (data → training → validation → inference → evaluation) has been
> run end to end, but only on CPU with small random-weight models. No video model has been trained, there are **no pretrained
> weights**, and GPU memory use / speed at full size have not been measured. Upstream DiT-SR checkpoints do **not** load.

**Contents**
[Overview](#overview) · [Requirements and installation](#requirements-and-installation) · [Quickstart (10 minutes, no data needed)](#quickstart) ·
[Prepare your data](#prepare-your-data) · [Scenarios: one command each](#scenarios-one-command-each) ·
[What you get](#what-you-get-outputs) · [Configuration reference](#configuration-reference) · [Inference in detail](#inference-in-detail) ·
[Evaluation](#evaluation) · [Image restoration](#image-restoration-upstream-pipeline) · [Troubleshooting](#troubleshooting) ·
[Repository layout](#repository-layout) · [Limitations](#known-limitations) · [Citation](#citation-and-acknowledgement)

**How the pieces fit together:** see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): flow charts of what calls what for training,
one training iteration, the model, validation, inference/sampling, patch aggregation and evaluation, plus the file hierarchy and a
per-scenario table.

---

## Overview

**Task.** Given a low-quality (LQ) video, produce a restored video that is `sf`× larger (4× or 2×) and temporally consistent.

**Method in one paragraph.** A conditional diffusion model works directly on RGB pixels (no autoencoder). The reverse process
starts from the bicubic-upsampled LQ video plus noise and refines it in a few steps (15 by default; ResShift-style
diffusion that shifts the residual between LQ and HR). The denoiser is a UNet whose blocks are Swin transformers conditioned on the
diffusion time step with **adaLN-Zero**, extended by **temporal attention** across frames. Training clips are degraded on the fly
with the Real-ESRGAN pipeline (blur, resize, noise, JPEG, second order). At inference, long videos are processed by overlapping
temporal windows and spatial tiles, or optionally by a WeatherDiff-style **patch aggregation** that keeps all patches consistent
at every diffusion step.

**What changed compared to DiT-SR**

| | DiT-SR (upstream) | This repo |
|---|---|---|
| Input | one image `[B,3,H,W]` | one image **or** N frames `[B,3,T,H,W]` |
| Space | latent (VQ autoencoder, f4/f8) | **pixel space**, no autoencoder |
| Time-step conditioning | AdaFM (FFT-domain scaling) | **adaLN-Zero** (shift/scale/gate, zero-initialised) |
| Temporal modelling | – | **temporal self-attention** in every Swin block |
| Data / trainer / sampler | images | images **and** videos |
| Long inputs | tiles | temporal windows × tiles, or patch aggregation |

Design details:

- **Frames are folded into the batch** for every per-frame layer (convs, window attention, MLP); only the temporal attention mixes
  frames. Its gate starts at 0, so a fresh model behaves like per-frame image SR. It also runs for `T=1`, so every parameter always
  takes part in training (multi-GPU safe) and one code path serves every `T`. `model.params.temporal_attn: False` removes it.
- **Pixel space without extra cost.** `DiTSRModel(patch_size=p)` pixel-unshuffles the input into channels (`p=4` for real-world
  SR, 8 for faces), runs the UNet at `H/p`, and shuffles back, i.e. the compute of the old latent UNet. The LQ condition is
  bicubic-resized to the HR grid inside the model, so `sf` can be 2 or 4 without changing the network.
- **Interface** (4× config): noisy HR clip `[B,3,T,H,W]`, LQ condition `[B,3,T,H/4,W/4]`, time steps `[B]`, output `[B,3,T,H,W]`.
  `H`, `W` must be multiples of `patch_size × 2^(levels-1) × window_size` (256 by default) unless patch aggregation is used.
- Change log and licence note: [`UPSTREAM.md`](UPSTREAM.md).

---

## Requirements and installation

- Linux, **an NVIDIA GPU with CUDA** (training and inference call `.cuda()`; there is no CPU mode), Python 3.10, PyTorch 2.1.1.
- GPU memory has not been measured. If you run out of memory use `small_gpu`, lower `train.microbatch`, or `--fp32` off/on as needed
  (see [Troubleshooting](#troubleshooting)).
- `ffmpeg` is optional (only to extract frames, see below).

```bash
git clone https://github.com/LiranAz/GenerativeVideoRestoration.git
cd GenerativeVideoRestoration

conda create -n gvr python=3.10 -y
conda activate gvr
pip install -r requirements.txt
```
The first run that builds LPIPS downloads the VGG weights (internet needed once).

---

## Quickstart

One command creates a synthetic dataset, trains a **tiny** model for 20 iterations (with validation), restores the validation
videos and evaluates them. It proves that your installation, GPU and the whole pipeline work; the result is not meant to look good.

```bash
bash scripts/quickstart.sh
```
What happens (each step is also a command you can run alone, see the next sections):

| Step | Command | Output |
|---|---|---|
| 1 | `scripts/run_scenario.sh demo-data` | `demo_data/{train,val/gt,val/lq,val/lq_x2}` (8 training videos, 2 validation videos) |
| 2 | `scripts/run_scenario.sh smoke --train demo_data/train --val_lq demo_data/val/lq --val_gt demo_data/val/gt --out runs/quickstart` | logs with `Train:` and `Validation Metric` lines, `runs/quickstart/<timestamp>/` |
| 3 | `scripts/run_scenario.sh infer --scenario smoke --ckpt <ema checkpoint> --input demo_data/val/lq --out runs/quickstart/results` | restored frames |
| 4 | `scripts/run_scenario.sh evaluate --sr runs/quickstart/results --gt demo_data/val/gt` | PSNR / SSIM / TDE table |

Everything below uses the same `scripts/run_scenario.sh` (run it without arguments, or with `help`, to see all options).

---

## Prepare your data

A **video** is either a **folder of frames** (sorted by file name: `00000.png`, `00001.png`, ...) or a **video file**
(`.mp4 .avi .mov .mkv .webm`). A dataset argument is a folder that contains videos; it is searched **recursively**.

```
train_videos/                      val_videos/
├── clip_a/        (frames)        ├── lq/                   LQ input, e.g. 1/4 resolution
│   ├── 00000.png                  │   ├── v1/ 00000.png ...
│   └── ...                        │   └── v2.mp4
├── clip_b.mp4                     └── gt/                   ground truth, same names as in lq/
└── group/clip_c/ ...                  ├── v1/ 00000.png ...
                                       └── v2.mp4
```

- **Training videos** are the *high-quality* originals. The low-quality version is synthesised on the fly (Real-ESRGAN degradation),
  so you only need good HR video, at least `gt_size` (256 px by default) on the shorter side (smaller ones are upscaled). Videos
  shorter than the clip length repeat their last frame.
- **Validation** (optional but recommended) needs LQ clips and, for PSNR/LPIPS, the matching GT clips with the **same names**.
  The LQ must be exactly `sf`× smaller than GT. Make them from GT frames, e.g. with ffmpeg:
  ```bash
  ffmpeg -i gt/v2.mp4 -vf "scale=iw/4:ih/4:flags=bicubic" lq/v2.mp4
  ```
- **Extract frames from a video** (a frame folder is lossless, mp4 is convenient):
  ```bash
  mkdir -p train_videos/clip_a && ffmpeg -i clip_a.mp4 train_videos/clip_a/%05d.png
  ```
- No data at hand? `scripts/run_scenario.sh demo-data` writes a synthetic set (not realistic; for testing the pipeline only).
- Public video SR datasets (for example REDS or Vimeo-90K) can be used by pointing `--train` at their frame folders; this repo does not
  download datasets.

---

## Scenarios: one command each

`scripts/run_scenario.sh <scenario> [options]` wraps `main.py` / `inference_video.py` / `evaluate_video.py`. It picks the right
config, passes your paths, and uses `torchrun` automatically when several GPUs are visible (`GPUS=n` to choose). Every scenario is a
config under `configs/examples/` that only lists what differs from `configs/vsr_DiT.yaml` (`_base_:` inheritance); any value can be
overridden on the command line (append `-- key=value ...`, i.e. `--set` in the raw commands).
Replace `<...>` with your paths. Quote paths that contain spaces or commas.

| # | Scenario | Config | Use it for | Cost |
|---|---|---|---|---|
| 1 | `smoke` | `configs/examples/vsr_smoke_test.yaml` | verify the setup | minutes, tiny model |
| 2 | `standard` | `configs/vsr_DiT.yaml` | real 4× video SR (256 px crops, 5 frames, 56M params) | days |
| 3 | `small_gpu` | `configs/examples/vsr_small_gpu.yaml` | 128 px crops, 3 frames, 20M params | smaller GPUs |
| 4 | `single_frame` | `configs/examples/vsr_single_frame.yaml` | image-style training from video frames (`T=1`) | like 2 |
| 5 | `x2` | `configs/examples/vsr_x2.yaml` | 2× instead of 4× | like 2 |
| 6 | resume / fine-tune | any of the above | continue a run / start from weights | – |
| 7 | `infer` | training config, or `vsr_patch_inference.yaml` | restore videos (tiled, or patch aggregation) | – |
| 8 | `evaluate` | – | PSNR / SSIM / LPIPS / temporal difference error | – |

**1. Smoke test**: check data paths, GPU, training, validation, checkpoints.
```bash
scripts/run_scenario.sh smoke --train <train_videos> --val_lq <val_lq> --val_gt <val_gt>
```
**2. Standard video SR (4×)**
```bash
scripts/run_scenario.sh standard --train <train_videos> --val_lq <val_lq> --val_gt <val_gt> --out runs/vsr
```
Defaults: `batch: [16, 1]`, `microbatch: 2` (gradient accumulation = batch/microbatch), 300k iterations, mixed precision.

**3. Small GPU**: crop 128, 3 frames, narrower model. The crop is one config value, `degradation.gt_size`; everything that depends
on it is derived and validated (see [Crop size](#crop-size)).
```bash
scripts/run_scenario.sh small_gpu --train <train_videos> --val_lq <val_lq> --val_gt <val_gt>
```
**4. Single frames (`T=1`)**: use when you only want per-frame restoration, or to pre-train before clips. The temporal layers still run
(attention over one frame), so the checkpoint can later be fine-tuned on clips (scenario 6).
```bash
scripts/run_scenario.sh single_frame --train <train_videos_or_frame_folders> --val_lq <val_lq> --val_gt <val_gt>
```
**5. 2× upscaling**: validation LQ must be 2× smaller than GT. LQ sizes are multiples of 128 px at 2× (64 at 4×).
```bash
scripts/run_scenario.sh x2 --train <train_videos> --val_lq <val_lq_x2> --val_gt <val_gt>
scripts/run_scenario.sh infer --scenario x2 --scale 2 --ckpt <ckpt> --input <lq_x2_video> --out results_x2
```
**6. Resume or fine-tune**
```bash
# continue the same run: restores model, EMA, learning-rate schedule and counters (use the same scenario/config)
scripts/run_scenario.sh standard --train <train_videos> --val_lq <val_lq> --val_gt <val_gt> --out runs/vsr \
    --resume runs/vsr/<timestamp>/ckpts/model_<iter>.pth
# start a NEW run from existing weights, e.g. single-frame -> clips
scripts/run_scenario.sh standard --train <train_videos> --val_lq <val_lq> --val_gt <val_gt> --out runs/finetune \
    --init runs/single_frame/<timestamp>/ckpts/model_<iter>.pth
```
**7. Inference.** Use the **same scenario/config as in training** (the crop and model size are part of the architecture) and preferably
the EMA weights (`ema_ckpts/ema_model_<iter>.pth`).
```bash
# tiled: temporal windows x spatial tiles, blended (works for any video size, padded internally)
scripts/run_scenario.sh infer --scenario standard --ckpt runs/vsr/<ts>/ema_ckpts/ema_model_<iter>.pth --input <video_or_frames_or_folder> --out results
# WeatherDiff-style patch aggregation: smoother, ~4x more compute, no tiling seams
scripts/run_scenario.sh infer --scenario standard --ckpt <ckpt> --input <...> --out results_patch --patch
```
The input can be a video file, a frame folder, or a folder of videos. Outputs: `.mp4` for video-file inputs, PNG frames for frame
folders (add `-- --save_frames` to also get PNGs for videos). Details: [Inference in detail](#inference-in-detail).

**8. Evaluate** (restored vs. ground truth, paired by name)
```bash
scripts/run_scenario.sh evaluate --sr results --gt <gt_videos> --metrics psnr,ssim,tde
```

**Without the wrapper.** `scripts/run_scenario.sh` only builds these commands:
```bash
python main.py --cfg_path configs/vsr_DiT.yaml --save_dir runs/vsr \
    --set "data.train.params.dir_paths=[<train_videos>]" data.val.params.lq_path=<val_lq> data.val.params.gt_path=<val_gt>
torchrun --standalone --nproc_per_node=<gpus> --nnodes=1 main.py ...            # several GPUs
python inference_video.py -i <video> -o results --config_path configs/vsr_DiT.yaml --ckpt_path <ckpt> [--patch_restoration true]
python evaluate_video.py -i results -r <gt_videos> --metrics psnr,ssim,tde --out_json metrics.json
```

---

## What you get (outputs)

```
runs/<name>/<timestamp>/
├── training.log          full config + loss / validation lines
├── ckpts/model_<iter>.pth        weights + counters (use for --resume)
├── ema_ckpts/ema_model_<iter>.pth   EMA weights (use for inference; best quality)
└── images/{train,val}/   lq / gt / diffused / x0-pred grids (training), sr / lq / gt grids (validation); one row per clip
```
Validation (every `train.val_freq` iterations) logs `PSNR` and `LPIPS` averaged over frames (without `gt_path` it only writes images).
Inference writes `<out>/<video name>.mp4` and/or `<out>/<video name>/<frame>.png`.

---

## Configuration reference

All keys live in `configs/vsr_DiT.yaml` (commented). The ones you will touch:

| Key | Default | Meaning |
|---|---|---|
| `degradation.gt_size` | 256 | **training crop** (HR px). One place; see [Crop size](#crop-size) |
| `degradation.sf` / `diffusion.params.sf` | 4 / 4 | upscaling factor (keep equal) |
| `degradation.queue_size` | 64 | training-pair pool (clips kept on the GPU for diversity); must be divisible by the batch size |
| `data.train.params.dir_paths` | placeholder | training videos |
| `data.train.params.num_frames` / `frame_stride` | 5 / `[1,2]` | temporal crop: frames per clip, random stride |
| `data.train.params.crop_type` | `random` | `random` or `center` spatial crop |
| `data.train.params.reverse_prob` | 0.1 | chance to play a clip backwards |
| `data.val.params.lq_path` / `gt_path` / `num_frames` | placeholder / placeholder / 5 | validation set (`gt_path: null` skips metrics) |
| `model.params.model_channels`, `swin_embed_dim`, `swin_depth` | 128, 160, 4 | model width / depth (56M params) |
| `model.params.patch_size` | 4 | pixel-unshuffle factor; `image_size` is derived from the crop (`~`) |
| `model.params.attention_resolutions` | `[64,32,16]` | folded UNet resolutions with transformer layers |
| `model.params.temporal_attn` | true | temporal attention on/off |
| `model.ckpt_path` | `~` | initialise from weights (fine-tuning) |
| `diffusion.params.steps` | 15 | diffusion steps (also the inference cost) |
| `train.lr`, `warmup_iterations`, `iterations` | 5e-5, 5000, 300000 | optimisation |
| `train.batch` | `[16, 1]` | `[training batch over all GPUs, validation batch]` |
| `train.microbatch` | 2 | clips per forward/backward; lower it for less memory |
| `train.save_freq` / `val_freq` | 10000 / same | checkpoint / validation interval (iterations) |
| `train.log_freq` | `[200, 2000, 1]` | `[loss, training images, validation images]`; must be ≥ 2 for the first entry |
| `train.use_amp`, `ema_rate` | true, 0.999 | mixed precision, EMA decay |
| `train.num_workers` | 4 | data loader processes (must be ≥ 1) |
| `patch_restoration.*` | disabled | [patch aggregation](#inference-in-detail) |

**Config inheritance and overrides.** A config may start with `_base_: configs/vsr_DiT.yaml`; keys are merged, lists replaced.
`--set key=value ...` overrides anything last, e.g. `--set train.iterations=1000 'data.train.params.dir_paths=[/data/a,/data/b]'`.

### Crop size
The training crop is the single value `degradation.gt_size` (HR pixels). At start-up `utils/util_crop.py` derives or checks everything
that depends on it: `model.params.image_size` (`gt_size / patch_size`), the dataset crop (`data.train.params.gt_size:
${degradation.gt_size}`), `patch_restoration.patch_size` and the LQ size step for validation and inference. The crop must be a
multiple of `patch_size × 2^(levels-1)` and every UNet level a multiple of `window_size` (or not larger than it); otherwise you get an
error that names the valid multiples (multiples of 256 always work with the default model, 128 works too). When you change the crop,
also adapt `attention_resolutions` (a warning appears if none matches), and remember that a checkpoint is only valid for the
crop and model size it was trained with.

---

## Inference in detail

`inference_video.py -i <video(s)> -o <out> --config_path <cfg> --ckpt_path <ckpt>`

- **Tiled (default).** The video is cut into temporal windows (`--num_frames`, default = training clip length; `--frame_overlap`, default
  2) and each window into overlapping spatial tiles (`--chop_size`, `--chop_stride`, in LQ pixels; the size must be a multiple of the LQ
  unit: 64 at 4×, 128 at 2×). Every frame starts from the same noise (a function of seed, frame index, position), and overlaps are blended
  with smooth weights, which keeps tiles and windows consistent.
- **Patch aggregation (optional, off by default).** Following [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion):
  the whole window is restored in *one* reverse process; at every step the model is applied to overlapping space-time patches of the
  current state, outputs are averaged per pixel, and one diffusion update is done on the full tensor, so patches can never drift apart.
  Any input size ≥ one patch works. Enable with `--patch_restoration true` (or `patch_restoration.enabled: True`, e.g.
  `configs/examples/vsr_patch_inference.yaml`):

  | key | default | meaning |
  |---|---|---|
  | `patch_size` | `~` (= crop) | spatial patch, HR px; must equal the training crop |
  | `patch_stride` | 128 | HR px; cost ≈ (patch_size / patch_stride)² model passes per pixel |
  | `patch_frames`, `frame_stride` | `~`, `~` | temporal patch length / stride (`~` = whole window / half of it) |
  | `weighting` | `uniform` | `tent` down-weights patch borders |
  | `batch_size` | 8 | patches per model call |

  The same block is used by validation inside training. Programmatic use:
  ```python
  from patch_restoration import PatchDiffusiveRestoration
  sr = PatchDiffusiveRestoration(base_diffusion, model.eval(), cfg_dict, sf=4).restore(lq)   # lq [B,3,T,h,w] in -1..1
  ```
- **Memory/speed knobs:** `--num_frames`, `--frame_overlap`, `--chop_size`, `--chop_stride` (tiled); `patch_stride`, `batch_size`, `patch_frames`
  (patch mode); `diffusion.params.steps`; `--fp32` disables mixed precision (slower, more memory).

---

## Evaluation

```bash
python evaluate_video.py -i results -r <gt_videos> --metrics psnr,ssim,tde --out_json metrics.json
```
Restored and ground-truth videos (files, frame folders, or folders of them) are paired by name (a single pair is paired regardless of
names). Per frame on the 8-bit Y channel, averaged: `psnr`, `ssim`, `lpips` (needs LPIPS weights), and `tde`, the *temporal difference
error* `mean |(SR_{t+1} − SR_t) − (GT_{t+1} − GT_t)|`, which grows with flicker and drift (lower is better). Use `--border N` to crop N pixels.

---

## Image restoration (upstream pipeline)

The original single-image pipeline still runs on the pixel-space model: `configs/realsr_DiT.yaml`, `realsr_DiT_Lite.yaml`, `faceir_DiT.yaml`
(training data lists in `data_list/*.txt`; images only, no video classes).
```bash
python main.py --cfg_path configs/realsr_DiT_Lite.yaml --save_dir runs/img      # train (edit the data paths in the config or use --set)
python inference.py --task realsr --scale 4 --chop_size 256 --chop_stride 224 \
    --config_path configs/realsr_DiT_Lite.yaml --ckpt_path <ckpt> -i <lq_images> -o results_img -r <gt_images>
```
`inference.py` ends with `evaluate.py`, which needs `pyiqa`, downloads metric weights and needs a GPU. `test_realsr.sh` / `test_faceir.sh`
refer to upstream checkpoints that do not match this architecture; pass your own.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CUDA out of memory` | lower `train.microbatch`; use `small_gpu`; fewer frames (`num_frames`); smaller `degradation.queue_size`. Inference: smaller `--chop_size`, `patch_restoration.batch_size`, or larger `patch_stride` |
| `ValueError: ... must be a multiple of ...` at start-up | the crop does not fit the UNet; use the multiple named in the message (`degradation.gt_size`) |
| `AssertionError: chop_size (..) must be a multiple of ..` | `--chop_size` must be a multiple of the LQ unit (64 at 4×, 128 at 2×) |
| `size mismatch` / missing keys when loading a checkpoint | use the exact config (crop, model size, `temporal_attn`) the checkpoint was trained with; upstream DiT-SR weights never match |
| `model.params.image_size is null` | build the model through `main.py`/`inference_video.py`, or call `utils.util_crop.resolve_crop(config)` first |
| `prefetch_factor option could only be specified in multiprocessing` | `train.num_workers` must be ≥ 1 |
| Loss logging crashes with `log_freq: [1, ...]` | the first `log_freq` entry must be ≥ 2 |
| `No videos found in ...` | check the path; frame folders need image files directly inside; video files need a supported extension |
| `--set` value is ignored / parse error | lists need brackets: `'x=[a,b]'`; quote paths with spaces; `null` sets a key to None |
| DDP "parameters not used in the loss" | only possible with `temporal_attn: True` and a modified model; the temporal branch runs for any `T` |
| Garbled/green frames from an `.mp4` | OpenCV could not decode the codec; re-encode (`ffmpeg -i in.mkv -c:v libx264 out.mp4`) or extract frames |
| Validation shows no PSNR/LPIPS | no `data.val.params.gt_path` (it is optional) |
| First start downloads files | LPIPS needs the VGG weights once (internet) |

---

## Repository layout

```
scripts/run_scenario.sh        one entry point for every scenario (train, infer, evaluate, demo data)
scripts/quickstart.sh          demo data -> smoke training -> inference -> evaluation
scripts/make_demo_data.py      synthetic demo dataset
main.py                        training entrypoint            inference_video.py   video inference
trainer_video.py / trainer.py  video / image trainers          sampler_video.py / sampler.py   video / image samplers
patch_restoration.py           WeatherDiff-style patch aggregation
evaluate_video.py              video metrics                   evaluate.py / inference.py     upstream image scripts
configs/vsr_DiT.yaml           base video config               configs/examples/   scenario configs
datapipe/video_datasets.py     video datasets                  datapipe/datasets.py   dataset factory
models/                        DiTSRModel (unet.py), Swin + temporal attention, diffusion
utils/                         util_config.py (inheritance, --set), util_crop.py (crop rules), helpers
docs/ARCHITECTURE.md           flow charts and file hierarchy
```

## Known limitations
- Untrained; quality, best `num_frames` / overlap / hyper-parameters are unknown. No pretrained weights.
- GPU memory and speed were not measured; multi-GPU (DDP) training was not run.
- Pixel-space diffusion is slower and harder to train than latent diffusion; memory grows with `T` (temporal attention is `O(T²)` per position).
- Temporal attention is only active at the `attention_resolutions` levels and in the middle block; other layers are per-frame.
- The degradation does not model temporal effects (video-codec artefacts, motion blur across frames); noise is independent per frame.
- Upstream DiT-SR ships no `LICENSE` file; check with its authors before redistributing or using this code beyond research.

## Citation and acknowledgement
This repo builds on DiT-SR; please cite it if you use this code:
```
@inproceedings{cheng2025effective,
  title={Effective diffusion transformer architecture for image super-resolution},
  author={Cheng, Kun and Yu, Lei and Tu, Zhijun and He, Xiao and Chen, Liyu and Guo, Yong and Zhu, Mingrui and Wang, Nannan and Gao, Xinbo and Hu, Jie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2455--2463},
  year={2025}
}
```
Patch aggregation follows the idea of WeatherDiffusion (Özdenizci & Legenstein, "Restoring Vision in Adverse Weather Conditions with
Patch-Based Denoising Diffusion Models"). Thanks also to [ResShift](https://github.com/zsyOAOA/ResShift), [DiT](https://github.com/facebookresearch/DiT),
[FFTFormer](https://github.com/kkkls/FFTformer), [SwinIR](https://github.com/JingyunLiang/SwinIR), [SinSR](https://github.com/wyf0912/SinSR)
and [BasicSR](https://github.com/XPixelGroup/BasicSR).
