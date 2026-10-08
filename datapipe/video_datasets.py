"""Video datasets.

Layout of every tensor returned by a dataset: ``[T, 3, H, W]`` (frame axis first). The default collate
turns that into ``[B, T, 3, H, W]`` and the trainer moves it to the model layout ``[B, 3, T, H, W]``.

A "video" is either
    * a folder that directly contains the frames (sorted by file name), or
    * a video file (.mp4, .avi, .mov, .mkv, .webm).
"""
import math
import random
import numpy as np
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from basicsr.data.realesrgan_dataset import RealESRGANDataset
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def is_video_file(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


def find_videos(roots, recursive=True):
    """Return a sorted list of video paths (frame folders and video files) found below `roots`."""
    roots = [roots] if isinstance(roots, (str, Path)) else roots
    found = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            found.append(root)
            continue
        candidates = [root] + (sorted(root.rglob('*')) if recursive else sorted(root.glob('*')))
        for path in candidates:
            if path.is_dir():
                if any(f.suffix.lower() in IMG_EXTS for f in path.iterdir()):
                    found.append(path)
            elif is_video_file(path):
                found.append(path)
    return sorted(set(found))


class VideoReader:
    """Random access to the frames of one video, as float32 RGB images in [0, 1] (h x w x 3)."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            self.frame_files = sorted(f for f in self.path.iterdir() if f.suffix.lower() in IMG_EXTS)
            self.num_frames = len(self.frame_files)
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(str(self.path))
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.frame_files = None
        if self.num_frames <= 0:
            raise IOError(f'No frames found in {self.path}')

    def read(self, indices):
        """Read frames at the given (sorted, possibly repeated) indices."""
        indices = [min(max(int(i), 0), self.num_frames - 1) for i in indices]
        frames = {}
        if self.cap is None:
            for i in set(indices):
                img = cv2.imread(str(self.frame_files[i]), cv2.IMREAD_COLOR)
                if img is None:
                    raise IOError(f'Cannot read {self.frame_files[i]}')
                frames[i] = img
        else:
            # one seek, then sequential decoding up to the last needed frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, min(indices))
            current = min(indices)
            wanted = set(indices)
            last = None
            while current <= max(indices):
                ok, img = self.cap.read()
                if not ok:
                    break
                last = img
                if current in wanted:
                    frames[current] = img
                current += 1
            for i in wanted:    # container reported more frames than it can decode: repeat the last one
                frames.setdefault(i, last)
            if last is None:
                raise IOError(f'Cannot decode {self.path}')
        return [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB).astype(np.float32) / 255. for i in indices]

    def read_all(self):
        return self.read(range(self.num_frames))


def to_tensor(frames):
    """list of h x w x 3 float arrays -> T x 3 x h x w tensor"""
    return torch.from_numpy(np.ascontiguousarray(np.stack(frames, 0).transpose(0, 3, 1, 2)))


class VideoRealESRGANDataset(RealESRGANDataset):
    """Training clips for Real-ESRGAN style synthetic degradation (done on GPU by the trainer).

    Returns ``gt`` of shape ``[T, 3, gt_size, gt_size]`` in [0, 1] together with ONE set of blur kernels per
    clip (the trainer applies the same kernels to all frames of a clip, so the degradation is temporally
    consistent). Spatial crop and horizontal flip are shared by all frames of a clip.

    Extra options (on top of the RealESRGANDataset ones):
        dir_paths / video_paths: folders to search for videos (see `find_videos`)
        num_frames:    T, frames per clip
        frame_stride:  [min, max] temporal stride, sampled uniformly per clip
        reverse_prob:  probability of playing the clip backwards
        gt_size:       spatial crop (HR pixels), identical for all frames of a clip
        crop_type:     'random' (default) or 'center'
    """

    def __init__(self, opt):
        opt = dict(opt)
        roots = opt.pop('dir_paths', None) or opt.pop('video_paths')
        opt.pop('txt_file_path', None)
        super().__init__(opt)    # builds the kernel settings; self.paths stays empty
        self.video_paths = find_videos(roots)
        if len(self.video_paths) == 0:
            raise ValueError(f'No videos found in {roots}')
        self.paths = self.video_paths
        self.num_frames = opt['num_frames']
        self.frame_stride = opt.get('frame_stride', [1, 1])
        self.reverse_prob = opt.get('reverse_prob', 0.0)
        self.gt_size = int(opt['gt_size'])
        self.crop_type = opt.get('crop_type', 'random')
        assert self.crop_type in ('random', 'center'), f"crop_type must be 'random' or 'center', got {self.crop_type}"

    def _sample_indices(self, n):
        T = self.num_frames
        stride = random.randint(*self.frame_stride)
        while stride > 1 and (T - 1) * stride >= n:
            stride -= 1
        span = (T - 1) * stride
        start = random.randint(0, max(n - 1 - span, 0))
        indices = [start + k * stride for k in range(T)]    # indices past the end are clamped by the reader
        if random.random() < self.reverse_prob:
            indices = indices[::-1]
        return indices

    def _sample_kernels(self):
        # same procedure as basicsr's RealESRGANDataset.__getitem__
        opt = self.opt
        kernel_size = random.choice(self.kernel_range1)
        if np.random.uniform() < opt['sinc_prob']:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list, self.kernel_prob, kernel_size, self.blur_sigma, self.blur_sigma,
                [-math.pi, math.pi], self.betag_range, self.betap_range, noise_range=None)
        pad_size = (self.blur_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < opt['sinc_prob2']:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2, self.kernel_prob2, kernel_size, self.blur_sigma2, self.blur_sigma2,
                [-math.pi, math.pi], self.betag_range2, self.betap_range2, noise_range=None)
        pad_size = (self.blur_kernel_size2 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        if np.random.uniform() < opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = torch.FloatTensor(circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.blur_kernel_size2))
        else:
            sinc_kernel = self.pulse_tensor
        return torch.FloatTensor(kernel), torch.FloatTensor(kernel2), sinc_kernel

    def __getitem__(self, index):
        path = self.video_paths[index]
        try:
            reader = VideoReader(path)
            frames = reader.read(self._sample_indices(reader.num_frames))
        except Exception:    # unreadable video: fall back to another one
            return self.__getitem__(random.randint(0, len(self) - 1))

        # make the shorter side at least gt_size (shared by all frames), then crop / flip identically
        h, w = frames[0].shape[:2]
        if min(h, w) < self.gt_size:
            ratio = self.gt_size / min(h, w)
            new_size = (math.ceil(w * ratio), math.ceil(h * ratio))
            frames = [cv2.resize(f, new_size, interpolation=cv2.INTER_CUBIC).clip(0, 1) for f in frames]
            h, w = frames[0].shape[:2]
        if self.crop_type == 'center':
            top, left = (h - self.gt_size) // 2, (w - self.gt_size) // 2
        else:
            top = random.randint(0, h - self.gt_size)
            left = random.randint(0, w - self.gt_size)
        frames = [f[top:top + self.gt_size, left:left + self.gt_size] for f in frames]
        if self.opt.get('use_hflip', True) and random.random() < 0.5:
            frames = [f[:, ::-1] for f in frames]

        kernel, kernel2, sinc_kernel = self._sample_kernels()
        return {'gt': to_tensor(frames), 'kernel1': kernel, 'kernel2': kernel2,
                'sinc_kernel': sinc_kernel, 'gt_path': str(path)}


class VideoPairedData(Dataset):
    """Validation clips: the first `num_frames` frames (after `start`) of every video.

    lq_path: videos (frame folders / files) with the low-quality input.
    gt_path: optional; a video with the same name for every lq video.
    Values are normalised with (x - mean) / std. Returned: {'lq': T x 3 x h x w, 'gt': ..., 'path': ...}
    """

    def __init__(self, lq_path, gt_path=None, num_frames=5, start=0, mean=0.5, std=0.5, need_path=False,
                 length=None):
        super().__init__()
        self.lq_videos = find_videos(lq_path)
        if length is not None:
            self.lq_videos = self.lq_videos[:length]
        self.gt_path = Path(gt_path) if gt_path is not None else None
        self.num_frames, self.start = num_frames, start
        self.mean, self.std = mean, std
        self.need_path = need_path

    def __len__(self):
        return len(self.lq_videos)

    def _load(self, path):
        reader = VideoReader(path)
        T = min(self.num_frames, reader.num_frames) if self.num_frames else reader.num_frames
        start = min(self.start, reader.num_frames - T)
        frames = to_tensor(reader.read(range(start, start + T)))
        return (frames - self.mean) / self.std

    def __getitem__(self, index):
        lq_video = self.lq_videos[index]
        out = {'lq': self._load(lq_video)}
        if self.gt_path is not None:
            candidates = [self.gt_path / lq_video.name, self.gt_path / lq_video.stem]
            gt_video = next((c for c in candidates if c.exists()), None)
            if gt_video is None:
                raise FileNotFoundError(f'No ground-truth video for {lq_video.name} in {self.gt_path}')
            out['gt'] = self._load(gt_video)
        if self.need_path:
            out['path'] = str(lq_video)
        return out
