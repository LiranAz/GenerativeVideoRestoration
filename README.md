<div align="center">
<h2>Generative Video Restoration</h2>
<p>Pixel-space video super-resolution with a diffusion transformer, forked from
<a href="https://github.com/kunncheng/DiT-SR">DiT-SR</a> (<a href="https://arxiv.org/abs/2409.19589">arXiv 2409.19589</a>).</p>
</div>

> **Status: research code, untrained.** The video pipeline has only been checked on CPU with small random-weight models
> (shapes, a few training steps, validation, windowed/tiled inference). No video model has been trained and there are no
> pretrained weights. Upstream DiT-SR checkpoints do **not** load (see [What changed](#what-changed-from-dit-sr)).

## What changed from DiT-SR

| | DiT-SR (upstream) | This repo |
|---|---|---|
| Input | one image `[B,3,H,W]` | one image **or** N frames `[B,3,T,H,W]` |
| Space | latent (VQ autoencoder, f4/f8) | **pixel space**, no autoencoder |
| Time-step conditioning | AdaFM (FFT-domain scaling) | **adaLN-Zero** (shift/scale/gate, zero-initialised) |
| Temporal modelling | – | **temporal self-attention** in every Swin block, gated by adaLN-Zero |
| Data / trainer / sampler | image only | image **and** video (`trainer_video.py`, `sampler_video.py`) |

Details:

- **Video model.** Frames are folded into the batch for all per-frame layers (convs, window attention, MLP). Each Swin
  block adds a temporal attention branch that attends across the T frames at every spatial position, with a sinusoidal
  frame-index embedding, so any T works at inference. The temporal gate starts at zero: at initialisation a video behaves
  exactly like per-frame image SR. `temporal_attn: False` in the model params disables it. The temporal branch also runs for `T=1` (single frames/images),
  where attention over one frame reduces to a value/output projection; this keeps all parameters in the graph, so training
  with `num_frames: 1` (or on single images) also works with multi-GPU DDP, and every `T` uses the same code path.
- **Pixel space.** To keep the compute of the old latent UNet, `DiTSRModel(patch_size=p)` pixel-unshuffles the input into
  channels (`p=4` for real-world SR, `p=8` for faces), runs the UNet at `image_size = H/p`, and shuffles the output back.
  The low-quality condition is bicubic-resized to the size of `x` inside the model.
- **Tensor shapes** (real-world SR config, `sf=4`, `p=4`):

  | Tensor | Shape |
  |---|---|
  | noisy HR video `x` | `[B, 3, T, H, W]` |
  | LQ condition `lq` | `[B, 3, T, H/4, W/4]` (same `T`) |
  | timesteps | `[B]` (shared by all frames) |
  | output | `[B, 3, T, H, W]` |

  `H` and `W` must be multiples of `patch_size × 8 × window_size` (256 for real-world SR, 512 for faces); pad otherwise.
  The training crop is configurable (`degradation.gt_size`, default 256; see below).
- Full change log and licence note: [`UPSTREAM.md`](UPSTREAM.md).

## Installation

```
git clone https://github.com/LiranAz/GenerativeVideoRestoration.git
cd GenerativeVideoRestoration

conda create -n gvr python=3.10 -y
conda activate gvr
pip install -r requirements.txt
```

`requirements.txt` is upstream's (torch 2.1.1, xformers optional). Training needs a CUDA GPU; LPIPS downloads VGG weights on
first use.

## Video restoration

### Data
A *video* is either a folder of frames (sorted by file name) or a video file (`.mp4 .avi .mov .mkv .webm`).
Edit `configs/vsr_DiT.yaml`:

```yaml
data:
  train:
    params:
      dir_paths: ['/path/to/train_videos']   # searched recursively
      num_frames: 5                          # T per training clip
      frame_stride: [1, 2]                   # temporal stride, random per clip
  val:
    params:
      lq_path: /path/to/val_videos/lq        # optional validation set
      gt_path: /path/to/val_videos/gt        # same video names as in lq_path
```

**Crop size.** The training crop is one config value, `degradation.gt_size` (HR pixels, default 256). Everything that depends
on it is derived or checked by `utils/util_crop.py` at start-up (`main.py`, `TrainerDifVSR`, `inference_video.py`):
`model.params.image_size` (`~` = auto, `gt_size / patch_size`), the dataset crop (`data.train.params.gt_size:
${degradation.gt_size}`), `patch_restoration.patch_size` (`~` = the crop) and the validation/inference LQ multiple
(`train.val_resolution: ~`). The crop must be a multiple of `patch_size * 2^(levels-1)` and every UNet level must be a
multiple of `window_size` (or not larger than it), otherwise you get an explanatory error: with the default model,
multiples of 256 always work, and e.g. 128 works too. `crop_type: random|center` selects where the crop is taken;
`num_frames` and `frame_stride` control the temporal crop. If you change the crop, also check
`model.params.attention_resolutions` (the resolutions of the folded UNet levels, i.e. `gt_size / patch_size / 2^k`;
a warning is raised if none matches), and note that a checkpoint is only valid for the crop it was trained with.

Training clips are cropped to `gt_size` and flipped identically for all frames. The Real-ESRGAN degradation is applied on
the GPU with **one set of blur kernels and one JPEG quality per clip**; resize factors and noise type are shared by the
batch, and noise is sampled independently for every frame.

### Example configs and run instructions
All example configs live in `configs/examples/` and only list what differs from `configs/vsr_DiT.yaml` (they start with
`_base_: configs/vsr_DiT.yaml`; lists are replaced, other keys are merged). Any config value can be overridden on the
command line with `--set key=value ...` (both `main.py` and `inference_video.py`), so you never have to edit a file
just to point at your data. Replace `<...>` placeholders; use `torchrun --standalone --nproc_per_node=<gpus> --nnodes=1`
instead of `python` for multi-GPU training.

| # | Scenario | Config | Use it for |
|---|---|---|---|
| 1 | Smoke test | `configs/examples/vsr_smoke_test.yaml` | check data, training, validation, checkpoints in minutes (tiny model, 20 iterations) |
| 2 | Standard video SR, 4x | `configs/vsr_DiT.yaml` | real training: 256 px crops, 5-frame clips, 56M parameters |
| 3 | Small GPU | `configs/examples/vsr_small_gpu.yaml` | 128 px crops, 3-frame clips, 20M parameters, gradient accumulation |
| 4 | Single frames | `configs/examples/vsr_single_frame.yaml` | image-style training from videos / frame folders (`T=1`) |
| 5 | 2x upscaling | `configs/examples/vsr_x2.yaml` | `sf=2` instead of `sf=4` |
| 6 | Resume / fine-tune | any of the above | continue a run, or start from a checkpoint |
| 7 | Inference | `configs/vsr_DiT.yaml` or `configs/examples/vsr_patch_inference.yaml` | restore videos, tiled or WeatherDiff-style patch aggregation |
| 8 | Evaluation | – | PSNR / SSIM / LPIPS / temporal difference error |

The data arguments used below (`dir_paths`, `lq_path`, `gt_path`) are searched recursively; a video is a folder of frames or
a video file. The validation set needs `lq_path`; `gt_path` is optional (`--set data.val.params.gt_path=null`: validation then only
writes restored images and skips PSNR/LPIPS).

**1. Smoke test** (tiny model, `use_amp: False`, so it also runs on small or older GPUs)
```
python main.py --cfg_path configs/examples/vsr_smoke_test.yaml --save_dir runs/smoke \
  --set "data.train.params.dir_paths=[<train_videos>]" data.val.params.lq_path=<val_lq> data.val.params.gt_path=<val_gt>
```
Expect `Train: ...`, `Validation Metric ...` lines and `runs/smoke/<timestamp>/{ckpts,ema_ckpts,images}`.

**2. Standard video SR (4x)**
```
torchrun --standalone --nproc_per_node=<gpus> --nnodes=1 main.py --cfg_path configs/vsr_DiT.yaml --save_dir runs/vsr \
  --set "data.train.params.dir_paths=[<train_videos>]" data.val.params.lq_path=<val_lq> data.val.params.gt_path=<val_gt>
```
Defaults: `batch: [16, 1]`, `microbatch: 2`, 300k iterations, mixed precision. Reduce `microbatch` if you run out of memory
(`batch / microbatch` = gradient accumulation steps).

**3. Small GPU (about 12-16 GB)**
```
python main.py --cfg_path configs/examples/vsr_small_gpu.yaml --save_dir runs/small \
  --set "data.train.params.dir_paths=[<train_videos>]" data.val.params.lq_path=<val_lq> data.val.params.gt_path=<val_gt>
```
The crop is changed in one place (`degradation.gt_size: 128`); the UNet resolution, dataset crop and patch size follow
(see *Crop size* above). `attention_resolutions` is adapted in the config because the folded UNet levels are 32/16/8/4.
Inference with a checkpoint from this config needs the same `--config_path` (the crop is part of the architecture).

**4. Single frames (`T=1`)**
```
python main.py --cfg_path configs/examples/vsr_single_frame.yaml --save_dir runs/single \
  --set "data.train.params.dir_paths=[<train_videos_or_frame_folders>]" data.val.params.lq_path=<val_lq> data.val.params.gt_path=<val_gt>
```
The temporal layers still run (attention over one frame) so the checkpoint can later be used on clips or fine-tuned with
scenario 2 via `model.ckpt_path`.

**5. 2x upscaling**
```
python main.py --cfg_path configs/examples/vsr_x2.yaml --save_dir runs/x2 \
  --set "data.train.params.dir_paths=[<train_videos>]" data.val.params.lq_path=<val_lq_x2> data.val.params.gt_path=<val_gt>
python inference_video.py -i <lq_x2_video> -o results_x2 --config_path configs/examples/vsr_x2.yaml \
  --ckpt_path runs/x2/<timestamp>/ckpts/model_<iter>.pth --scale 2 --chop_size 128 --chop_stride 96
```
At 2x the LQ multiple is 128 pixels, so `--chop_size` must be a multiple of 128 (64 at 4x) and validation LQ clips are cropped
to multiples of 128.

**6. Resume or fine-tune**
```
# continue the same run (restores model, EMA, learning rate schedule and log counters; keep the same config)
python main.py --cfg_path configs/vsr_DiT.yaml --resume runs/vsr/<timestamp>/ckpts/model_<iter>.pth
# start a NEW run from existing weights (e.g. single-frame -> clips)
python main.py --cfg_path configs/vsr_DiT.yaml --save_dir runs/finetune \
  --set model.ckpt_path=runs/single/<timestamp>/ckpts/model_<iter>.pth "data.train.params.dir_paths=[<train_videos>]"
```
`--resume` takes the checkpoint path and finds `ema_ckpts/ema_<name>.pth` next to it.

**7. Inference**
```
# tiled (default): temporal windows x spatial tiles, blended
python inference_video.py -i <video_or_frames_or_folder> -o results --config_path configs/vsr_DiT.yaml \
  --ckpt_path runs/vsr/<timestamp>/ema_ckpts/ema_model_<iter>.pth
# WeatherDiff-style patch aggregation (smoother, ~4x more compute; any size >= one patch)
python inference_video.py -i <video_or_frames_or_folder> -o results_patch --config_path configs/examples/vsr_patch_inference.yaml \
  --ckpt_path runs/vsr/<timestamp>/ema_ckpts/ema_model_<iter>.pth
# same thing without a separate config
python inference_video.py -i <...> -o results_patch --config_path configs/vsr_DiT.yaml --ckpt_path <...> --patch_restoration true
```
Use the EMA weights (`ema_ckpts/ema_model_<iter>.pth`) for best results; plain checkpoints are `ckpts/model_<iter>.pth`.
Memory knobs: `--chop_size/--chop_stride` (tiled mode), `--num_frames/--frame_overlap` (both modes),
`patch_restoration.batch_size` / `patch_stride` (patch mode). Add `--fp32` to disable mixed precision.

**8. Evaluation** (restored videos against ground truth, paired by name)
```
python evaluate_video.py -i results -r <gt_videos> --metrics psnr,ssim,tde --out_json metrics.json
```

### Train
```
torchrun --standalone --nproc_per_node=<gpus> --nnodes=1 main.py --cfg_path configs/vsr_DiT.yaml --save_dir ${save_dir}
```
Checkpoints go to `${save_dir}/ckpts` (and EMA weights to `${save_dir}/ema_ckpts`). `batch`, `microbatch` and
`degradation.queue_size` in the config are untuned guesses; the training-pair pool stores whole clips, so lower
`queue_size` if memory is tight. Add `--resume` to continue from `save_dir`.

### Inference
```
python inference_video.py -i input.mp4 -o results --ckpt_path ${save_dir}/ckpts/model_xxx.pth --config_path configs/vsr_DiT.yaml
```
`-i` can be a video file, a folder of frames, or a folder of videos. Output is an `.mp4` (add `--save_frames` for PNG
frames; frame folders are always written as PNGs).

Long videos are handled with sliding temporal windows (`--num_frames`, `--frame_overlap`) and overlapping spatial tiles
(`--chop_size`, `--chop_stride`, in LQ pixels; `chop_size` must be a multiple of the LQ unit: 64 at 4x, 128 at 2x). Overlaps are blended with smooth
weights, and the initial noise of every frame depends only on the seed, the frame index and the pixel position, so
overlapping windows and tiles start from identical noise and agree with each other.

### Evaluate
```
python evaluate_video.py -i results/ -r gt_videos/ --metrics psnr,ssim,tde --out_json metrics.json
```
Restored and ground-truth videos (files, frame folders, or folders of them) are paired by name. Metrics are computed per
frame on the Y channel (8-bit) and averaged: `psnr`, `ssim`, `lpips` (needs the LPIPS weights) and `tde`, the
*temporal difference error* `mean |(SR_{t+1}-SR_t) - (GT_{t+1}-GT_t)|`, which grows with flicker and temporal drift.
`evaluate.py` is the upstream image-only script (no-reference metrics via `pyiqa`, needs a GPU and downloads weights).

### Optional: WeatherDiff-style patch-based restoration
Instead of restoring tiles independently, the whole clip can be restored in **one** reverse process in which, at every
step, the model is applied to overlapping space-time patches of the current `x_t`, the outputs are averaged per pixel and
one diffusion update is done on the full tensor (idea from [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion);
implementation in `patch_restoration.py`, class `PatchDiffusiveRestoration` / wrapper `PatchAggregatedModel`). Neighbouring
patches then stay consistent at every step, so there are no tiling seams, and any input size >= `patch_size` works.

It is **off by default**; enable it in `configs/vsr_DiT.yaml`:
```yaml
patch_restoration:
  enabled: True
  patch_size: 256      # HR pixels; must equal the training crop (degradation.gt_size)
  patch_stride: 128    # HR pixels; patch_size // 2 = 4x the cost of non-overlapping patches
  patch_frames: ~      # temporal patch length (~ = all frames of the clip)
  frame_stride: ~      # temporal stride (~ = patch_frames // 2)
  weighting: uniform   # uniform (WeatherDiff) | tent (down-weights patch borders)
  batch_size: 8        # patches per model call
```
or per run with `python inference_video.py ... --patch_restoration true`. The block is also used by validation in
`trainer_video.py`. When enabled, `--chop_size/--chop_stride` are ignored; the temporal windows are still used for long
videos. Programmatic use:
```python
from patch_restoration import PatchDiffusiveRestoration
restorer = PatchDiffusiveRestoration(base_diffusion, model.eval(), cfg_dict, sf=4)
sr = restorer.restore(lq)          # lq: [B,3,T,h,w] (or [B,3,h,w]) in [-1, 1]
```
The cost grows with the overlap: patches per pixel = (patch_size / patch_stride)² (x the temporal overlap).

## Image restoration (upstream pipeline, still available)

The original single-image pipeline works with the pixel-space model; `configs/realsr_DiT.yaml`, `configs/realsr_DiT_Lite.yaml`
and `configs/faceir_DiT.yaml` are kept. You have to train from scratch (upstream weights are for the latent model).

Training data of upstream: [LSDIR](https://huggingface.co/ofsoundof/LSDIR), [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/),
[DIV8K](https://ieeexplore.ieee.org/document/9021973), [OutdoorSceneTraining](https://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/),
[Flickr2K](https://www.kaggle.com/datasets/hliang001/flickr2k) and the first 10K [FFHQ](https://github.com/NVlabs/ffhq-dataset)
faces; the image paths are listed in `data_list/*.txt`.

```
# real-world image SR
torchrun --standalone --nproc_per_node=8 --nnodes=1 main.py --cfg_path configs/realsr_DiT.yaml --save_dir ${save_dir}
# blind face restoration
torchrun --standalone --nproc_per_node=8 --nnodes=1 main.py --cfg_path configs/faceir_DiT.yaml --save_dir ${save_dir}
# inference (edit the checkpoint path in the scripts)
bash test_realsr.sh
bash test_faceir.sh
```
`test_realsr.sh` / `test_faceir.sh` point at the upstream checkpoints `weights/realsr.pth` / `weights/faceir.pth`, which do not
match this architecture; pass your own trained checkpoint instead.

## Repository layout

```
utils/util_crop.py          crop-size resolution/validation (single source: degradation.gt_size)
models/unet.py              DiTSRModel (5-D input, pixel-unshuffle stem)
models/swin_transformer.py  adaLN-Zero Swin block + TemporalAttention
models/gaussian_diffusion.py  ResShift-style diffusion (pixel space, video-aware)
datapipe/video_datasets.py  clip datasets (frame folders / video files)
trainer.py                  image trainers (Real-ESRGAN degradation in TrainerDifIR._degrade)
trainer_video.py            TrainerDifVSR: video training and validation
sampler.py / inference.py   image sampler and CLI
sampler_video.py / inference_video.py   windowed + tiled video sampler and CLI
evaluate_video.py           PSNR / SSIM / LPIPS / temporal-difference error for videos
patch_restoration.py        optional WeatherDiff-style per-step patch aggregation
configs/                    vsr_DiT.yaml (video), realsr_*.yaml, faceir_DiT.yaml
```

## Known limitations
- Untrained; quality, optimal `num_frames`/overlap and hyper-parameters are unknown.
- Pixel-space diffusion is slower and harder to train than the latent version; memory grows with `T` (temporal attention is
  `O(T²)` per position).
- Temporal attention is only active at the `attention_resolutions` levels and in the middle block; other layers are per-frame.
- The degradation pipeline does not model temporal effects (e.g. video-codec artefacts or motion blur across frames).
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
Thanks to the projects DiT-SR builds on: [ResShift](https://github.com/zsyOAOA/ResShift), [DiT](https://github.com/facebookresearch/DiT),
[FFTFormer](https://github.com/kkkls/FFTformer), [SwinIR](https://github.com/JingyunLiang/SwinIR),
[SinSR](https://github.com/wyf0912/SinSR) and [BasicSR](https://github.com/XPixelGroup/BasicSR).
