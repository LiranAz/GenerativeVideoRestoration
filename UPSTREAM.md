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
