"""WeatherDiff-style patch-based diffusive restoration (optional).

Idea (Ozdenizci & Legenstein, "Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising
Diffusion Models", https://github.com/IGITUGraz/WeatherDiffusion, MIT): the model is trained on small random
patches, and at inference the reverse process runs on the FULL image/clip. At every reverse step the model is
evaluated on a grid of overlapping patches of the current x_t, the predictions are averaged per pixel, and one
diffusion update is done on the whole tensor. All patches therefore stay consistent at every step, which
avoids the seams of "restore tiles independently, stitch at the end".

Here this is implemented as a wrapper module with the signature of DiTSRModel, so the existing diffusion
code is reused untouched:

    wrapper = PatchAggregatedModel(model, patch_size=256, patch_stride=128, ...)
    out = base_diffusion.p_sample_loop(y=lq, model=wrapper, model_kwargs={'lq': lq}, ...)

Averaging the raw model output is valid for every `predict_type` (xstart, epsilon, residual, ...) because the
diffusion turns it into an x_0 estimate / posterior mean per pixel. Patches are space-time blocks
[T_p x p x p], so overlapping frames agree as well. Enabled through the `patch_restoration` block of the
config (see configs/vsr_DiT.yaml); with `enabled: False` nothing changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_ops import frames_to_batch, batch_to_frames

DEFAULTS = dict(
    enabled=False,
    patch_size=256,      # spatial patch, in HR pixels; must equal the training crop (gt_size)
    patch_stride=128,    # spatial stride, in HR pixels (patch_size // 2 -> 4x the cost of non-overlapping tiles)
    patch_frames=None,   # temporal patch length; None -> all frames of the clip
    frame_stride=None,   # temporal stride; None -> patch_frames // 2 (at least 1)
    weighting='uniform', # 'uniform' (WeatherDiff) or 'tent' (down-weight patch borders)
    batch_size=8,        # patches per model call
)


def get_patch_cfg(configs):
    """Merge configs.patch_restoration with DEFAULTS. Returns a plain dict (enabled flag included)."""
    cfg = dict(DEFAULTS)
    user = configs.get('patch_restoration', None) if configs is not None else None
    if user is not None:
        cfg.update({k: v for k, v in dict(user).items() if k in DEFAULTS})
    return cfg


def grid_starts(total, size, stride, align=1):
    """Start offsets (multiples of `align`) such that windows of `size` cover [0, total) completely.

    Unlike WeatherDiff's `range(0, total - size + 1, r)`, the last window is always shifted to the border,
    so every pixel is covered even when (total - size) is not a multiple of the stride.
    """
    assert total >= size, f'cannot place a window of {size} in {total}'
    stride = max(align, (stride // align) * align)
    starts = list(range(0, total - size + 1, stride))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts


def _tent(n):
    u = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    return 1.0 - (2 * u - 1).abs()      # in (0, 1], max at the centre


class PatchAggregatedModel(nn.Module):
    """Wraps a restoration model so it can be applied to inputs larger than its training patch."""

    def __init__(self, model, patch_size=256, patch_stride=128, patch_frames=None, frame_stride=None,
                 weighting='uniform', batch_size=8):
        super().__init__()
        assert weighting in ('uniform', 'tent'), weighting
        self.model = model
        self.patch_size, self.patch_stride = patch_size, patch_stride
        self.patch_frames, self.frame_stride = patch_frames, frame_stride
        self.weighting, self.batch_size = weighting, batch_size
        self._cache = {}

    # ------------------------------------------------------------------ grid
    def _grid(self, T, H, W, scale):
        p = self.patch_size
        assert p % scale == 0, f'patch_size {p} must be a multiple of the LQ scale {scale}'
        Tp = T if self.patch_frames is None else min(self.patch_frames, T)
        fs = self.frame_stride if self.frame_stride is not None else max(Tp // 2, 1)
        ts = grid_starts(T, Tp, fs)
        ys = grid_starts(H, p, self.patch_stride, align=scale)
        xs = grid_starts(W, p, self.patch_stride, align=scale)
        return Tp, [(t, y, x) for t in ts for y in ys for x in xs]

    def _weight(self, Tp, p, device):
        key = (Tp, p, self.weighting, str(device))
        if key not in self._cache:
            if self.weighting == 'tent':
                wt = _tent(Tp) if Tp > 1 else torch.ones(1)
                w = wt[:, None, None] * _tent(p)[None, :, None] * _tent(p)[None, None, :]
            else:
                w = torch.ones(Tp, p, p)
            self._cache[key] = w.to(device)
        return self._cache[key]

    # --------------------------------------------------------------- forward
    def forward(self, x, timesteps, lq=None, mask=None):
        """x: [B,C,T,H,W] (or [B,C,H,W]); lq (and mask): same layout at LQ resolution (H/s, W/s)."""
        is_video = x.ndim == 5
        if not is_video:
            x = x[:, :, None]
            lq = lq[:, :, None] if lq is not None else None
            mask = mask[:, :, None] if mask is not None else None
        B, C, T, H, W = x.shape
        p = self.patch_size
        scale = H // lq.shape[-2] if lq is not None else 1
        assert lq is None or (H == lq.shape[-2] * scale and W == lq.shape[-1] * scale), 'lq/x size mismatch'
        if H < p or W < p:      # smaller than one patch: nothing to aggregate, plain model call
            out = self.model(x if is_video else x[:, :, 0], timesteps, **(
                {k: (v if is_video else v[:, :, 0]) for k, v in (('lq', lq), ('mask', mask)) if v is not None}))
            return out
        Tp, coords = self._grid(T, H, W, scale)
        w = self._weight(Tp, p, x.device)                                   # Tp x p x p
        q = p // scale

        out = torch.zeros(B, self._out_channels(x), T, H, W, device=x.device, dtype=torch.float32)
        wsum = torch.zeros(1, 1, T, H, W, device=x.device, dtype=torch.float32)
        for c0 in range(0, len(coords), self.batch_size):
            chunk = coords[c0:c0 + self.batch_size]
            xb = torch.cat([x[:, :, t:t + Tp, y:y + p, xx:xx + p] for t, y, xx in chunk], dim=0)
            kwargs = {}
            if lq is not None:
                kwargs['lq'] = torch.cat(
                    [lq[:, :, t:t + Tp, y // scale:y // scale + q, xx // scale:xx // scale + q] for t, y, xx in chunk], 0)
            if mask is not None:
                kwargs['mask'] = torch.cat(
                    [mask[:, :, t:t + Tp, y // scale:y // scale + q, xx // scale:xx // scale + q] for t, y, xx in chunk], 0)
            pred = self.model(xb, timesteps.repeat(len(chunk)), **kwargs).float()
            for (t, y, xx), pr in zip(chunk, pred.split(B, dim=0)):
                out[:, :, t:t + Tp, y:y + p, xx:xx + p] += pr * w
                wsum[:, :, t:t + Tp, y:y + p, xx:xx + p] += w
        out = out / wsum
        out = out.to(x.dtype)
        return out if is_video else out[:, :, 0]

    @staticmethod
    def _out_channels(x):
        return x.shape[1]       # diffusion models here predict in the same space as x_t


def wrap_model(model, patch_cfg):
    """Return `model` unchanged, or a PatchAggregatedModel when patch_cfg['enabled']."""
    if not patch_cfg.get('enabled', False):
        return model
    return PatchAggregatedModel(
        model, patch_size=patch_cfg['patch_size'], patch_stride=patch_cfg['patch_stride'],
        patch_frames=patch_cfg['patch_frames'], frame_stride=patch_cfg['frame_stride'],
        weighting=patch_cfg['weighting'], batch_size=patch_cfg['batch_size'])


class PatchDiffusiveRestoration:
    """Counterpart of WeatherDiff's `DiffusiveRestoration`: restores a full image/clip with patch aggregation.

    diffusion: GaussianDiffusion built by the config; model: trained DiTSRModel (eval mode).
    """

    def __init__(self, diffusion, model, patch_cfg=None, sf=4):
        self.diffusion, self.sf = diffusion, sf
        self.patch_cfg = {**DEFAULTS, **(patch_cfg or {}), 'enabled': True}
        self.model = wrap_model(model, self.patch_cfg)

    @torch.no_grad()
    def restore(self, lq, noise=None, clip_denoised=True):
        """lq: [B,3,T,h,w] or [B,3,h,w] in [-1, 1] -> restored tensor at sf x resolution, same layout."""
        is_video = lq.ndim == 5
        sf, p = self.sf, self.patch_cfg['patch_size']
        h, w = lq.shape[-2:]
        # patch-based restoration works for any size >= one patch; smaller inputs are reflect-padded
        ph, pw = max(0, -(-p // sf) - h), max(0, -(-p // sf) - w)
        if ph or pw:
            x4, T = frames_to_batch(lq)
            lq = batch_to_frames(F.pad(x4, (0, pw, 0, ph), mode='reflect'), T, is_video)
            if noise is not None:    # keep the given noise, fresh noise only for the padded border
                full = torch.randn(*noise.shape[:-2], (h + ph) * sf, (w + pw) * sf, device=noise.device, dtype=noise.dtype)
                full[..., :h * sf, :w * sf] = noise
                noise = full
        out = self.diffusion.p_sample_loop(
            y=lq, model=self.model, noise=noise, clip_denoised=clip_denoised,
            model_kwargs={'lq': lq}, device=lq.device, progress=False)
        return out[..., :h * sf, :w * sf]
