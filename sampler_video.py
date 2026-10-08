"""Video super-resolution sampler: sliding temporal windows x overlapping spatial tiles, blended.

Every window of `num_frames` frames and every spatial tile goes through the model as ONE clip, so the
temporal attention sees all frames of the window. To keep the result consistent across windows and
tiles, the initial diffusion noise of each frame is a function of (seed, global frame index, pixel
position) only: overlapping windows/tiles start from identical noise, and overlapping predictions are
averaged with smooth weights.
"""
import math
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sampler import BaseSampler
from datapipe.video_datasets import VideoReader, find_videos, is_video_file, IMG_EXTS


def _blend_weights(length, overlap, floor=0.1):
    """1-D weights: ramp up/down over `overlap` samples at both ends, never below `floor`."""
    w = torch.ones(length)
    ov = min(overlap, length // 2)
    if ov > 0:
        ramp = torch.linspace(floor, 1.0, ov + 1)[1:] if ov > 0 else torch.ones(0)
        w[:ov] = torch.minimum(w[:ov], ramp)
        w[-ov:] = torch.minimum(w[-ov:], ramp.flip(0))
    return w


def _starts(total, size, stride):
    """Start offsets covering [0, total) with windows of `size`; the last one is shifted to fit."""
    if total <= size:
        return [0]
    starts = list(range(0, total - size + 1, stride))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts


class VideoSampler(BaseSampler):
    def __init__(
            self, configs, sf=4, use_amp=True,
            chop_size=128, chop_stride=96,         # spatial tiles, in LQ pixels
            num_frames=None, frame_overlap=2,       # temporal window / overlap, in frames
            padding_offset=64, seed=10000,
            ):
        if num_frames is None:
            num_frames = configs.get('data', {}).get('train', {}).get('params', {}).get('num_frames', 5)
        assert chop_size % padding_offset == 0, \
            f'chop_size ({chop_size}) must be a multiple of {padding_offset}'
        assert 0 <= frame_overlap < num_frames
        self.num_frames, self.frame_overlap = num_frames, frame_overlap
        super().__init__(configs, sf=sf, use_amp=use_amp, chop_size=chop_size, chop_stride=chop_stride,
                         padding_offset=padding_offset, seed=seed)

    # ------------------------------------------------------------------ noise
    def _frame_noise(self, frame_idx, height, width, device):
        g = torch.Generator(device=device)
        g.manual_seed(self.seed * 1_000_003 + int(frame_idx))
        return torch.randn(3, height, width, generator=g, device=device)

    # ------------------------------------------------------------------- clip
    @torch.no_grad()
    def sample_clip(self, lq, frame_ids):
        """
        lq: 1 x 3 x T x h x w in [-1, 1] (h, w multiples of padding_offset)
        frame_ids: global index of every frame (for the noise)
        returns 1 x 3 x T x h*sf x w*sf in [-1, 1]
        """
        _, _, T, h, w = lq.shape
        sf, dev = self.sf, lq.device
        ch = min(self.chop_size, h)
        cw = min(self.chop_size, w)
        stride = self.chop_stride
        overlap = max(ch - stride, 0)
        ys, xs = _starts(h, ch, stride), _starts(w, cw, stride)

        noise = torch.stack([self._frame_noise(i, h * sf, w * sf, dev) for i in frame_ids], dim=1)[None]
        out = torch.zeros(1, 3, T, h * sf, w * sf, device=dev)
        wsum = torch.zeros(1, 1, 1, h * sf, w * sf, device=dev)
        wy = _blend_weights(ch * sf, overlap * sf).to(dev)
        wx = _blend_weights(cw * sf, overlap * sf).to(dev)
        weight = (wy[:, None] * wx[None, :])[None, None, None]

        context = torch.cuda.amp.autocast if self.use_amp else nullcontext
        for y in ys:
            for x in xs:
                tile = lq[..., y:y + ch, x:x + cw]
                tile_noise = noise[..., y * sf:(y + ch) * sf, x * sf:(x + cw) * sf]
                with context():
                    sr = self.base_diffusion.p_sample_loop(
                            y=tile, model=self.model, noise=tile_noise, clip_denoised=True,
                            model_kwargs={'lq': tile} if self.configs.model.params.cond_lq else None,
                            progress=False,
                            )
                sl = (..., slice(y * sf, (y + ch) * sf), slice(x * sf, (x + cw) * sf))
                out[sl] += sr.float().clamp(-1, 1) * weight
                wsum[sl] += weight
        return out / wsum

    # ------------------------------------------------------------------ video
    @torch.no_grad()
    def sample_video(self, read_frames, num_total):
        """
        Generator over (frame_index, sr_frame) in order. sr_frame: 3 x H*sf x W*sf float in [0, 1] (cpu).
        read_frames(indices) -> list of h x w x 3 float32 RGB arrays in [0, 1].
        """
        W = min(self.num_frames, num_total)
        step = max(W - self.frame_overlap, 1)
        starts = _starts(num_total, W, step)
        acc, wacc = {}, {}
        twin = _blend_weights(W, self.frame_overlap, floor=0.2)
        next_out = 0
        for k, s in enumerate(tqdm(starts, desc='windows')):
            ids = list(range(s, s + W))
            frames = read_frames(ids)
            h0, w0 = frames[0].shape[:2]
            lq = torch.from_numpy(np.stack(frames, 0)).permute(3, 0, 1, 2)[None].cuda()   # 1 x 3 x T x h x w
            lq = lq * 2 - 1
            off = self.padding_offset
            ph, pw = (-h0) % off, (-w0) % off
            if ph or pw:
                lq = F.pad(lq.flatten(1, 2), (0, pw, 0, ph), mode='reflect').unflatten(1, (3, W))
            sr = self.sample_clip(lq, ids)[..., :h0 * self.sf, :w0 * self.sf]
            sr = (sr[0] * 0.5 + 0.5).clamp(0, 1).cpu()                                     # 3 x T x H x W
            for t, i in enumerate(ids):
                acc[i] = acc.get(i, 0) + sr[:, t] * twin[t]
                wacc[i] = wacc.get(i, 0) + twin[t]
            # frames before the next window are final
            final_until = starts[k + 1] if k + 1 < len(starts) else num_total
            while next_out < final_until:
                yield next_out, acc.pop(next_out) / wacc.pop(next_out)
                next_out += 1

    def inference(self, in_path, out_path, save_frames=False, fps=None):
        """in_path: video file or folder of frames (a folder of videos is processed video by video)."""
        in_path, out_path = Path(in_path), Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        videos = find_videos(in_path)
        self.write_log(f'Found {len(videos)} video(s) in {in_path}')
        for video in videos:
            reader = VideoReader(video)
            name = video.stem if video.is_file() else video.name
            self.write_log(f'{name}: {reader.num_frames} frames')
            writer = None
            frame_dir = out_path / name
            if save_frames or video.is_dir():
                frame_dir.mkdir(parents=True, exist_ok=True)
            for i, sr in self.sample_video(reader.read, reader.num_frames):
                img = (sr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)[..., ::-1]   # RGB -> BGR
                if writer is None and is_video_file(video):
                    H, W = img.shape[:2]
                    writer = cv2.VideoWriter(str(out_path / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps or reader.fps, (W, H))
                if writer is not None:
                    writer.write(np.ascontiguousarray(img))
                if save_frames or video.is_dir():
                    stem = reader.frame_files[i].stem if reader.frame_files else f'{i:06d}'
                    cv2.imwrite(str(frame_dir / f'{stem}.png'), np.ascontiguousarray(img))
            if writer is not None:
                writer.release()
