"""Single source of truth for the training crop.

The crop size is `degradation.gt_size` (HR pixels). Everything that depends on it is derived here, so changing
the crop is a one-line config edit:

  * model.params.image_size   (UNet resolution = crop / patch_size)    -> set when null, checked otherwise
  * data.train.params.gt_size (dataset crop)                           -> must equal the crop
  * patch_restoration.patch_size (inference patch)                     -> crop when null
  * train.val_resolution      (validation crops, LQ pixels)            -> `lq_multiple(configs)` when null

Sizes must fit the UNet: the folded resolution `crop / patch_size` has to be divisible by 2^(levels-1), and the
resolution of every level has to be a multiple of `window_size` (or not larger than it).
"""
import warnings


def _model_geometry(configs):
    mp = configs.model.params
    return mp.get('patch_size', 1), len(mp.channel_mult), mp.window_size


def hr_multiple(configs):
    """Smallest HR size step for which the UNet accepts arbitrary multiples (patch * 2^(L-1) * window)."""
    p, levels, window = _model_geometry(configs)
    return p * 2 ** (levels - 1) * window


def lq_multiple(configs):
    """hr_multiple expressed in LQ pixels (what inference / validation have to pad or crop LQ to)."""
    sf = configs.diffusion.params.get('sf', 1)
    unit = hr_multiple(configs)
    assert unit % sf == 0, f'HR step {unit} not divisible by sf={sf}'
    return unit // sf


def check_hr_size(configs, size, what='size'):
    """Raise ValueError unless the UNet can run on an HR input of `size` x `size` pixels."""
    p, levels, window = _model_geometry(configs)
    if size % p != 0:
        raise ValueError(f'{what}={size} must be a multiple of model patch_size={p}')
    image_size = size // p
    if image_size % 2 ** (levels - 1) != 0:
        raise ValueError(f'{what}={size} / patch_size {p} = {image_size} is not divisible by 2^(levels-1) = '
                         f'{2 ** (levels - 1)}; use a multiple of {p * 2 ** (levels - 1)}')
    ds_levels = [image_size // 2 ** k for k in range(levels)]
    bad = [ds for ds in ds_levels if ds > window and ds % window != 0]
    if bad:
        raise ValueError(f'{what}={size}: UNet resolutions {ds_levels} must be multiples of window_size={window} '
                         f'(or <= it); {bad} are not. Use a size that is a multiple of {hr_multiple(configs)}.')


def resolve_crop(configs):
    """Fill / validate all crop-dependent values in place. No-op for configs without degradation.gt_size."""
    if configs.get('degradation', None) is None or configs.degradation.get('gt_size', None) is None:
        return configs
    crop = int(configs.degradation.gt_size)
    mp = configs.model.params
    p, levels, window = _model_geometry(configs)

    check_hr_size(configs, crop, 'crop (degradation.gt_size)')
    image_size = crop // p
    if mp.get('image_size', None) is None:
        mp.image_size = image_size
    elif int(mp.image_size) != image_size:
        raise ValueError(f'model.params.image_size={mp.image_size} does not match crop {crop} / patch_size {p} = '
                         f'{image_size}; set image_size to null (auto) or fix gt_size')
    ds_levels = [image_size // 2 ** k for k in range(levels)]
    att = list(mp.get('attention_resolutions', []))
    if att and not any(ds in att for ds in ds_levels):
        warnings.warn(f'None of the UNet resolutions {ds_levels} is in model.params.attention_resolutions={att}: '
                      f'only the middle block will use (temporal) transformer layers. Adjust attention_resolutions '
                      f'when changing the crop.')

    train = configs.get('data', {}).get('train', None) if configs.get('data', None) is not None else None
    if train is not None and train.get('params', None) is not None:
        gt = train.params.get('gt_size', None)
        if gt is None:
            train.params.gt_size = crop
        elif int(gt) != crop:
            raise ValueError(f'data.train.params.gt_size={gt} differs from degradation.gt_size={crop}; '
                             f'use gt_size: ${{degradation.gt_size}} in the data block')

    pr = configs.get('patch_restoration', None)
    if pr is not None and pr.get('patch_size', None) is None:
        pr.patch_size = crop
    return configs
