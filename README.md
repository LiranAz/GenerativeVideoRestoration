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
  exactly like per-frame image SR. `temporal_attn: False` in the model params disables it.
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
(`--chop_size`, `--chop_stride`, in LQ pixels; `chop_size` must be a multiple of 64). Overlaps are blended with smooth
weights, and the initial noise of every frame depends only on the seed, the frame index and the pixel position, so
overlapping windows and tiles start from identical noise and agree with each other.

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
