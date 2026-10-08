"""Trainer for video super-resolution (see configs/vsr_DiT.yaml).

Differences to trainer.TrainerDifIR (which it extends):
  * clips of T frames: batches are [B, 3, T, H, W] (gt) and [B, 3, T, H/sf, W/sf] (lq),
  * the Real-ESRGAN degradation is applied to the frames folded into the batch, with the same blur
    kernels and the same JPEG quality for all frames of a clip,
  * validation samples whole clips and reports per-frame PSNR / LPIPS.
"""
import math
import torch
import torch.nn.functional as F
from einops import rearrange

from trainer import TrainerDifIR
from utils import util_image


class TrainerDifVSR(TrainerDifIR):
    # ------------------------------------------------------------------ data
    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """Training pool (as in TrainerDifIR) but for tensors of any rank; the first axis is the batch."""
        b = self.lq.shape[0]
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b * 10)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, *self.lq.shape[1:]).cuda()
            self.queue_gt = torch.zeros(self.queue_size, *self.gt.shape[1:]).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:    # pool is full: swap in the new clips, take out random old ones
            idx = torch.randperm(self.queue_size)
            self.queue_lr, self.queue_gt = self.queue_lr[idx], self.queue_gt[idx]
            lq_dequeue, gt_dequeue = self.queue_lr[:b].clone(), self.queue_gt[:b].clone()
            self.queue_lr[:b], self.queue_gt[:b] = self.lq.clone(), self.gt.clone()
            self.lq, self.gt = lq_dequeue, gt_dequeue
        else:
            self.queue_lr[self.queue_ptr:self.queue_ptr + b] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b] = self.gt.clone()
            self.queue_ptr += b

    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        if phase == 'train' and self.configs.data.train.type == 'video_realesrgan':
            gt = data['gt'].cuda()                                    # B x T x 3 x H x W, [0, 1]
            B, T = gt.shape[:2]
            gt = gt.flatten(0, 1)                                     # (B*T) x 3 x H x W
            kernels = [data[k].cuda().repeat_interleave(T, dim=0) for k in ('kernel1', 'kernel2', 'sinc_kernel')]
            im_gt, im_lq = self._degrade(gt, *kernels, clip_len=T)
            im_gt, im_lq = (im_gt - 0.5) / 0.5, (im_lq - 0.5) / 0.5   # [-1, 1]

            # drop clips that contain NaNs
            im_lq = im_lq.reshape(B, T, *im_lq.shape[1:])
            im_gt = im_gt.reshape(B, T, *im_gt.shape[1:])
            bad = torch.isnan(im_lq).flatten(1).any(dim=1)
            if bad.any():
                with open(f"records_nan_rank{self.rank}.log", 'a') as f:
                    f.write(f'Find Nan value in rank{self.rank}\n')
                assert (~bad).any(), 'all clips of the batch contain NaN'
                im_lq, im_gt = im_lq[~bad], im_gt[~bad]

            # B x T x C x H x W -> B x C x T x H x W (model layout)
            self.lq = im_lq.permute(0, 2, 1, 3, 4).contiguous()
            self.gt = im_gt.permute(0, 2, 1, 3, 4).contiguous()
            self._dequeue_and_enqueue()
            return {'lq': self.lq.contiguous(), 'gt': self.gt}
        elif phase == 'val':
            # crop lq to a multiple of val_resolution and gt to the matching sf * size
            offset = self.configs.train.get('val_resolution', 64)
            sf = self.configs.diffusion.params.sf
            lq = data['lq'].permute(0, 2, 1, 3, 4)                    # B x C x T x h x w
            h, w = lq.shape[-2:]
            if h >= offset and w >= offset:
                h_end, w_end = (h // offset) * offset, (w // offset) * offset
                lq = lq[..., :h_end, :w_end]
            else:
                mode = self.configs.train.get('val_padding_mode', 'reflect')
                h_end, w_end = math.ceil(h / offset) * offset, math.ceil(w / offset) * offset
                lq = F.pad(lq.flatten(1, 2), (0, w_end - w, 0, h_end - h), mode=mode).unflatten(1, (lq.shape[1], lq.shape[2]))
            out = {'lq': lq}
            if 'gt' in data:
                gt = data['gt'].permute(0, 2, 1, 3, 4)
                out['gt'] = gt[..., :h_end * sf, :w_end * sf]
            return {k: v.cuda().to(dtype=dtype).contiguous() for k, v in out.items()}
        else:
            return {k: v.cuda().to(dtype=dtype) for k, v in data.items() if torch.is_tensor(v)}

    # --------------------------------------------------------------- logging
    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        """5-D clips [B, C, T, H, W] are logged as a grid with one row per clip."""
        if im_tensor.ndim == 5:
            T = im_tensor.shape[2]
            im_tensor = rearrange(im_tensor, 'b c t h w -> (b t) c h w')
            nrow = T
        super().logging_image(im_tensor, tag, phase, add_global_step=add_global_step, nrow=nrow)

    # ------------------------------------------------------------ validation
    def validation(self, phase='val'):
        if self.rank != 0:
            return
        if self.configs.train.use_ema_val:
            self.reload_ema_model()
            model = self.ema_model.eval()
        else:
            model = self.model.eval()

        num_clips = num_frames = 0
        mean_psnr = mean_lpips = 0.0
        num_iters_epoch = math.ceil(len(self.datasets[phase]) / self.configs.train.batch[1])
        for ii, data in enumerate(self.dataloaders[phase]):
            data = self.prepare_data(data, phase='val')
            im_lq = data['lq']
            model_kwargs = {'lq': im_lq} if self.configs.model.params.cond_lq else None
            with torch.no_grad():
                im_sr = self.base_diffusion.p_sample_loop(
                        y=im_lq, model=model, clip_denoised=True,
                        model_kwargs=model_kwargs, device=f"cuda:{self.rank}", progress=False,
                        ).clamp(-1.0, 1.0)

            if 'gt' in data:
                im_gt = data['gt']
                sr_f = rearrange(im_sr, 'b c t h w -> (b t) c h w')
                gt_f = rearrange(im_gt, 'b c t h w -> (b t) c h w')
                mean_psnr += util_image.batch_PSNR(
                        sr_f * 0.5 + 0.5, gt_f * 0.5 + 0.5, ycbcr=self.configs.train.val_y_channel)
                with torch.no_grad():
                    mean_lpips += self.lpips_loss(sr_f, gt_f).sum().item()
                num_frames += sr_f.shape[0]
            num_clips += im_sr.shape[0]

            if (ii + 1) % self.configs.train.log_freq[2] == 0:
                self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')
                self.logging_image(im_sr, tag='sr', phase=phase, add_global_step=False)
                if 'gt' in data:
                    self.logging_image(im_gt, tag='gt', phase=phase, add_global_step=False)
                self.logging_image(im_lq, tag='lq', phase=phase, add_global_step=True)

        if num_frames > 0:
            mean_psnr /= num_frames
            mean_lpips /= num_frames
            self.logger.info(f'Validation Metric (per frame, {num_clips} clips): '
                             f'PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}...')
            self.logging_metric(mean_psnr, tag='PSNR', phase=phase, add_global_step=False)
            self.logging_metric(mean_lpips, tag='LPIPS', phase=phase, add_global_step=True)
        self.logger.info("=" * 100)

        if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
            self.model.train()
