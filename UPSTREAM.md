# Upstream

This repository was initialized from [DiT-SR](https://github.com/kunncheng/DiT-SR)
(Cheng et al., "DiT-SR: Shifting Diffusion Transformer for Image Super-Resolution", NeurIPS 2024),
commit `43fc6f464dcebb0a7c24df2c11331db1c972b539` ("Update README.md").

The upstream repository ships no LICENSE file; check with the authors before redistributing
or relying on this code beyond research use. The original README is kept unchanged as `README.md`.

## Changes from upstream

- `models/swin_transformer.py`: AdaFM (FFT-domain scaling) replaced by adaLN-Zero (DiT-style
  per-channel shift/scale/gate from the timestep embedding, zero-initialised so each block starts as identity).
  Block renamed `SwinTransformerBlock_AdaLNZero`. **Upstream pretrained checkpoints are not compatible**
  (`adaLN_scale_*` keys replaced by `adaLN_modulation`), so models must be trained from scratch.
- Video input: `DiTSRModel` (and `GaussianDiffusion`) accept `[B, C, T, H, W]` as well as
  `[B, C, H, W]`. Frames are folded into the batch for all per-frame layers; each Swin block gains a
  temporal self-attention branch (across T at every spatial location, sinusoidal frame-index embedding, adaLN-Zero
  gated) so any T works. At init the temporal branch is off, so a video behaves like per-frame image SR.
  `temporal_attn: False` in the model params disables it.
- Pixel space: the VQ autoencoder is removed (configs, trainers, sampler, `ldm` VAE code, `first_stage_model` plumbing).
  Diffusion runs on RGB directly (`latent_flag: False`). To keep the compute of the old f4/f8 latent UNet,
  `DiTSRModel` takes `patch_size` (4 for realsr, 8 for faceir): pixels are pixel-unshuffled into channels before
  the UNet (`image_size = H / patch_size`) and shuffled back after. The LQ condition is bicubic-resized to x's
  grid inside the model. Pretrained upstream weights are not compatible.
- Video pipeline: `datapipe/video_datasets.py` (clip datasets), `trainer_video.py` (`TrainerDifVSR`, config
  `configs/vsr_DiT.yaml`), `sampler_video.py` / `inference_video.py` (windowed + tiled inference). The image
  trainer's Real-ESRGAN degradation was factored into `TrainerDifIR._degrade` (behaviour unchanged) so both share it.
- `patch_restoration.py`: optional patch-based diffusive restoration following the idea of
  [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion) (MIT, Ozdenizci & Legenstein): model outputs are
  averaged over overlapping patches at every reverse step. Re-implemented (not copied) as a wrapper model, so it works
  with this repo's diffusion and `predict_type`s and extends to space-time patches; the grid always covers the borders.
  Controlled by the `patch_restoration` block of `configs/vsr_DiT.yaml` (disabled by default).
