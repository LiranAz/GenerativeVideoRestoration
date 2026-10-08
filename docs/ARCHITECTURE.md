# Architecture: what calls what

Flow charts are [Mermaid](https://mermaid.js.org/) diagrams (rendered by GitHub; any Mermaid viewer works). Box text is
`file::function_or_class`. Arrows mean "calls / hands data to". Start with section 1 to find your scenario, then follow
the chart of the phase you care about.

Contents: 1 [Scenario map](#1-scenario-map) · 2 [Config resolution](#2-config-resolution) · 3 [Training](#3-training-flow) ·
4 [One training iteration](#4-one-training-iteration) · 5 [Model forward](#5-model-hierarchy-and-forward-pass) ·
6 [Validation](#6-validation) · 7 [Inference / sampling](#7-inference-and-sampling) ·
8 [Patch aggregation](#8-patch-aggregation-weatherdiff-style) · 9 [Evaluation](#9-evaluation) ·
10 [Image pipeline](#10-image-pipeline-upstream-still-available) · 11 [Scenario differences](#11-what-each-scenario-changes) ·
12 [File hierarchy](#12-file-hierarchy) · 13 [Tensor layouts](#13-tensor-layouts-cheat-sheet)

---

## 1. Scenario map
Everything starts either from `scripts/run_scenario.sh` (a thin wrapper) or directly from one of four entrypoints.
A *scenario* is just a config (`_base_` inheritance) plus the entrypoint that consumes it.

```mermaid
flowchart LR
    U(["user"]) --> RS["scripts/run_scenario.sh"]
    U --> QS["scripts/quickstart.sh"]
    QS --> RS
    RS -->|demo-data| DD["scripts/make_demo_data.py"]
    RS -->|"smoke, standard, small_gpu, single_frame, x2"| MAIN["main.py  (train + validate)"]
    RS -->|infer| INF["inference_video.py"]
    RS -->|evaluate| EV["evaluate_video.py"]

    MAIN --> C1["configs/examples/vsr_smoke_test.yaml"]
    MAIN --> C2["configs/vsr_DiT.yaml"]
    MAIN --> C3["configs/examples/vsr_small_gpu.yaml"]
    MAIN --> C4["configs/examples/vsr_single_frame.yaml"]
    MAIN --> C5["configs/examples/vsr_x2.yaml"]
    INF --> C2
    INF --> C6["configs/examples/vsr_patch_inference.yaml"]
    C1 & C3 & C4 & C5 & C6 -->|"_base_"| C2

    MAIN -.->|"writes"| CK[("runs/&lt;name&gt;/&lt;timestamp&gt;/ckpts, ema_ckpts, images, training.log")]
    CK -.->|"--ckpt_path"| INF
    INF -.->|"writes"| OUT[("restored .mp4 / PNG frames")]
    OUT -.-> EV
```

| Scenario | Command (`scripts/run_scenario.sh ...`) | Entrypoint | Config |
|---|---|---|---|
| Smoke test | `smoke` | `main.py` | `configs/examples/vsr_smoke_test.yaml` |
| Standard 4x video SR | `standard` | `main.py` | `configs/vsr_DiT.yaml` |
| Small GPU | `small_gpu` | `main.py` | `configs/examples/vsr_small_gpu.yaml` |
| Single frames (T=1) | `single_frame` | `main.py` | `configs/examples/vsr_single_frame.yaml` |
| 2x upscaling | `x2` | `main.py` | `configs/examples/vsr_x2.yaml` |
| Resume / fine-tune | `<scenario> --resume F` / `--init F` | `main.py` | same as the scenario |
| Inference (tiled) | `infer ...` | `inference_video.py` | training config of the checkpoint |
| Inference (patch aggregation) | `infer ... --patch` | `inference_video.py` | + `patch_restoration.enabled` |
| Evaluation | `evaluate ...` | `evaluate_video.py` | – |
| Image SR (upstream) | – | `main.py` / `inference.py` | `configs/realsr_*.yaml`, `configs/faceir_DiT.yaml` |

## 2. Config resolution
Done at the start of every entrypoint. This is where a scenario's overrides and the crop-dependent values are fixed.

```mermaid
flowchart TD
    A["CLI: --cfg_path / --config_path  +  --set key=value ..."] --> B["utils/util_config.py::load_config"]
    B -->|"follows _base_ recursively, merges, applies --set"| C["OmegaConf config"]
    C --> D["utils/util_crop.py::resolve_crop"]
    D --> D1["check_hr_size: crop must fit the UNet (patch_size * 2^(levels-1), window_size)"]
    D --> D2["model.params.image_size = gt_size / patch_size  (when null)"]
    D --> D3["data.train.params.gt_size must equal degradation.gt_size"]
    D --> D4["patch_restoration.patch_size = crop  (when null)"]
    D --> E["resolved config"]
    E --> F["lq_multiple(config): LQ size step used by validation and inference padding"]
```

## 3. Training flow
`python main.py --cfg_path <config>` (or `torchrun ... main.py` for several GPUs).

```mermaid
flowchart TD
    M["main.py"] --> L["load_config + resolve_crop  (section 2)"]
    L --> T["trainer_video.py::TrainerDifVSR.__init__ -> TrainerBase.__init__: setup_dist, setup_seed"]
    T --> TR["TrainerDifVSR.train()   defined in trainer.py::TrainerBase.train"]
    TR --> S1["1. init_logger: save_dir/timestamp with ckpts, ema_ckpts, images, training.log"]
    S1 --> S2["2. build_model"]
    S2 --> S2a["TrainerBase.build_model: DiTSRModel from config, load model.ckpt_path if set, DDP wrap, EMA copy"]
    S2 --> S2b["TrainerDifIR.build_model: LPIPS net, base_diffusion = script_util.py::create_gaussian_diffusion"]
    S2 --> S3["3. setup_optimizaton: AdamW, optional cosine scheduler, AMP scaler"]
    S3 --> S4["4. resume_from_ckpt: only with --resume (model, EMA, lr schedule, log counters)"]
    S4 --> S5["5. build_dataloader: datapipe/datasets.py::create_dataset"]
    S5 --> DS1["train: video_realesrgan -> VideoRealESRGANDataset"]
    S5 --> DS2["val (rank 0): video_paired -> VideoPairedData"]
    S5 --> LOOP{{"6. for iteration in start .. train.iterations"}}
    LOOP --> P["prepare_data(batch)"]
    P --> TS["training_step(data)   (section 4)"]
    TS --> V{"iteration % val_freq == 0 ?"}
    V -->|yes| VAL["validation()   (section 6)"]
    V -->|no| LR["adjust_lr: warm-up, then cosine if configured"]
    VAL --> LR
    LR --> SV{"iteration % save_freq == 0 ?"}
    SV -->|yes| CK["save_ckpt: ckpts/model_N.pth and ema_ckpts/ema_model_N.pth"]
    SV -->|no| LOOP
    CK --> LOOP
```

**Datasets in detail**

```mermaid
flowchart TD
    A["VideoRealESRGANDataset.__getitem__(index)"] --> B["find_videos / VideoReader: frame folder or video file, random access to frames"]
    B --> C["_sample_indices: T frames, random temporal stride, optional reversal"]
    C --> D["resize if shorter side &lt; gt_size, then ONE crop (random/center) + ONE flip shared by all T frames"]
    D --> E["_sample_kernels: ONE set of blur / sinc kernels for the clip"]
    E --> F["returns gt [T,3,H,W] in 0..1, kernel1, kernel2, sinc_kernel"]
    F --> G["DataLoader collate: gt [B,T,3,H,W]"]
```
The degraded LQ clip is **not** made in the dataset: it is made on the GPU in `prepare_data` (next section).

## 4. One training iteration

```mermaid
flowchart TD
    A["batch from DataLoader: gt [B,T,3,H,W] + kernels"] --> B["TrainerDifVSR.prepare_data (phase=train)"]
    B --> B1["fold frames into batch: gt [B*T,3,H,W]; kernels repeat_interleave T"]
    B1 --> B2["trainer.py::TrainerDifIR._degrade   (Real-ESRGAN pipeline on GPU)"]
    B2 --> B2a["blur(kernel1) - random resize - noise - JPEG  (first order)"]
    B2a --> B2b["blur(kernel2) - resize - noise  (second order, random)"]
    B2b --> B2c["resize to H/sf + sinc filter + JPEG (order random)  -  JPEG quality shared per clip via _jpeg_quality"]
    B2c --> B2d["paired_random_crop to gt_size"]
    B2d --> B3["normalise to -1..1, drop NaN clips, permute to [B,3,T,H,W]"]
    B3 --> B4["_dequeue_and_enqueue: training-pair pool for more diversity"]
    B4 --> C["TrainerDifIR.training_step(data)"]
    C --> C1["split into micro-batches (train.microbatch); gradient accumulation = batch/microbatch"]
    C1 --> C2["t ~ uniform over diffusion steps"]
    C2 --> D["models/gaussian_diffusion.py::GaussianDiffusion.training_losses(model, gt, lq, t)"]
    D --> D1["upsample_lq: y = bicubic(lq, x sf)"]
    D1 --> D2["q_sample: x_t = x0 + eta_t (y - x0) + kappa sqrt(eta_t) noise"]
    D2 --> D3["model(_scale_input(x_t), t, lq=lq)   -> models/unet.py::DiTSRModel.forward  (section 5)"]
    D3 --> D4["loss = MSE(prediction, x0)   (predict_type: xstart)"]
    D4 --> E["backward_step: loss.backward() (AMP scaler if enabled)"]
    E --> F["log_step_train: loss per timestep bucket, images every log_freq"]
    F --> G["optimizer.step (AdamW)"]
    G --> H["update_ema_model: EMA of the weights (ema_rate)"]
```

## 5. Model hierarchy and forward pass
`models/unet.py::DiTSRModel` takes an image `[B,3,H,W]` or a clip `[B,3,T,H,W]`.

```mermaid
flowchart TD
    A["DiTSRModel.forward(x, t, lq)"] --> B["basic_ops.py::frames_to_batch: [B,3,T,H,W] -> [B*T,3,H,W]"]
    B --> C["lq: bicubic-resize to x size, then F.pixel_unshuffle(p)"]
    B --> D["x: F.pixel_unshuffle(p):  [B*T,3p^2,H/p,W/p]"]
    A --> E["time embedding: timestep_embedding -> MLP -> emb, repeated per frame"]
    C --> F["concat(x, lq) = 6p^2 channels"]
    D --> F
    F --> G["input_blocks: conv, then per level [BasicLayer (if level in attention_resolutions)] and Downsample"]
    G --> H["middle_block: BasicLayer"]
    H --> I["output_blocks: concat skip connection, conv, BasicLayer, Upsample"]
    I --> J["out: GroupNorm, SiLU, conv -> 3p^2 channels"]
    J --> K["F.pixel_shuffle(p), then batch_to_frames: [B,3,T,H,W]"]
    E -.-> G
    E -.-> H
    E -.-> I
```

**Inside one `BasicLayer`** (`models/swin_transformer.py`), repeated `swin_depth` times, each block a `SwinTransformerBlock_AdaLNZero`:

```mermaid
flowchart TD
    X["features [B*T,C,h,w] + time embedding emb"] --> M["adaLN_modulation(emb): SiLU + Linear, zero-initialised -> shift/scale/gate for 3 branches"]
    X --> S["1. spatial: norm1 * (1+scale) + shift, window_partition (+ cyclic shift), WindowAttention, window_reverse, x + gate * out"]
    M -.-> S
    S --> T["2. temporal: norm_t * (1+scale) + shift, TemporalAttention over the T frames at every pixel, x + gate * out   (also runs for T=1)"]
    M -.-> T
    T --> F["3. MLP: norm2 * (1+scale) + shift, Mlp, x + gate * out"]
    M -.-> F
    F --> O["features [B*T,C,h,w]"]
```
Because every gate starts at 0, a fresh model is an identity map in all three branches (adaLN-Zero); temporal attention is
the only place frames interact (convs, window attention and MLP see frames independently).

## 6. Validation
Runs inside training (`val_freq`) on rank 0, using EMA weights when `train.use_ema_val`.

```mermaid
flowchart TD
    A["TrainerDifVSR.validation"] --> B["reload_ema_model, model.eval"]
    B --> B2["patch_restoration.py::wrap_model: PatchAggregatedModel if patch_restoration.enabled, else the model unchanged"]
    B2 --> C["for each batch of the val DataLoader (VideoPairedData: lq clip, gt clip)"]
    C --> D["prepare_data(phase=val): lq -> [B,3,T,h,w], crop lq to a multiple of lq_multiple, gt to sf x that"]
    D --> E["GaussianDiffusion.p_sample_loop(y=lq, model, model_kwargs lq)   (section 7 inner loop)"]
    E --> F["clamp to -1..1"]
    F --> G["PSNR (Y channel) + LPIPS per frame vs gt  (skipped when gt_path is null)"]
    F --> H["logging_image: sr / gt / lq grids, one row per clip"]
    G --> I["log mean PSNR / LPIPS, restore model.train()"]
```

## 7. Inference and sampling
`python inference_video.py -i <video> --config_path <cfg> --ckpt_path <ckpt>`

```mermaid
flowchart TD
    A["inference_video.py::main"] --> B["load_config + resolve_crop; padding_offset = lq_multiple(config)"]
    B --> C["sampler_video.py::VideoSampler.__init__"]
    C --> C1["sampler.py::BaseSampler.__init__: build_model (diffusion from config, DiTSRModel + checkpoint, eval, frozen)"]
    C --> C2["patch_restoration.enabled ?  PatchDiffusiveRestoration(diffusion, model)   (section 8)"]
    A --> D["VideoSampler.inference(in_path, out_path)"]
    D --> E["datapipe/video_datasets.py::find_videos + VideoReader  (file, frame folder, or folder of videos)"]
    E --> F["VideoSampler.sample_video: sliding temporal windows (num_frames, frame_overlap)"]
    F --> G["per window: read frames, pad LQ to a multiple of padding_offset (tiled mode only)"]
    G --> H["VideoSampler.sample_clip(window)"]
    H --> I{"patch restorer enabled?"}
    I -->|no: tiled| J["spatial tiles (chop_size, chop_stride) with per-frame deterministic noise"]
    J --> K["per tile: GaussianDiffusion.p_sample_loop(tile, model)"]
    K --> L["blend tiles with smooth weights"]
    I -->|yes| M["patch_restorer.restore(window)   one reverse process over the whole window"]
    L --> N["blend overlapping windows (triangular temporal weights); emit frames once final"]
    M --> N
    N --> O["write .mp4 (cv2.VideoWriter) and/or PNG frames"]
```

**The diffusion sampling loop** (identical for validation, tiles and patches; `models/gaussian_diffusion.py`):

```mermaid
flowchart TD
    A["p_sample_loop(y, model, noise)"] --> B["p_sample_loop_progressive"]
    B --> C["z_y = upsample_lq(y)  (bicubic x sf)"]
    C --> D["x_T = prior_sample: z_y + kappa * sqrt(eta_T) * noise"]
    D --> E{{"for i = steps-1 ... 0"}}
    E --> F["p_sample(model, x, z_y, t=i)"]
    F --> G["p_mean_variance: model(_scale_input(x), t, lq=lq)  -> predicted x0"]
    G --> H["q_posterior_mean_variance: mean and variance of x_{t-1} given x_t and predicted x0"]
    H --> I["x_{t-1} = mean + sqrt(variance) * noise   (no noise at the last step)"]
    I --> E
    E -->|done| J["x_0  =  restored frames"]
```
`model` here is either the plain `DiTSRModel` or the `PatchAggregatedModel` wrapper; the loop does not know the difference.

## 8. Patch aggregation (WeatherDiff-style)
Optional, off by default (`patch_restoration.enabled`). Idea from [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion):
instead of restoring tiles independently, aggregate overlapping patch predictions at **every** reverse step.

```mermaid
flowchart TD
    A["PatchDiffusiveRestoration.restore(lq clip)"] --> B["reflect-pad if the clip is smaller than one patch"]
    B --> C["GaussianDiffusion.p_sample_loop(y=lq, model=PatchAggregatedModel)"]
    C --> D{{"each reverse step"}}
    D --> E["PatchAggregatedModel.forward(x_t full, t, lq)"]
    E --> F["_grid: space-time patch positions (patch_size, patch_stride, patch_frames, frame_stride); border always covered"]
    F --> G["crop x_t and lq for a batch of patches (batch_size)"]
    G --> H["DiTSRModel on the patch batch"]
    H --> I["scatter-add weighted outputs (uniform or tent) into a full-size buffer"]
    I --> J["divide by the summed weights -> one full-size prediction"]
    J --> K["back in p_sample: ONE posterior update on the whole tensor"]
    K --> D
```

## 9. Evaluation
`python evaluate_video.py -i <restored> -r <ground truth>`

```mermaid
flowchart TD
    A["evaluate_video.py::main"] --> B["pair_videos: match restored and reference by name (file stem / folder name)"]
    B --> C["evaluate_pair(sr, gt)"]
    C --> D["VideoReader.read: frames of both videos, crop to the common size"]
    D --> E["to_y: 8-bit Y channel (BT.601)"]
    E --> F["psnr per frame"]
    E --> G["ssim per frame (skimage)"]
    E --> H["tde: mean |diff_t(SR) - diff_t(GT)|  temporal difference error"]
    D --> I["lpips per frame (optional, needs the LPIPS weights)"]
    F & G & H & I --> J["per-video and mean table, optional --out_json"]
```

## 10. Image pipeline (upstream, still available)
Same trainer/diffusion/model code, single images, no video classes.

```mermaid
flowchart LR
    A["main.py + configs/realsr_DiT*.yaml or faceir_DiT.yaml"] --> B["trainer.py::TrainerDifIR"]
    B --> C["basicsr RealESRGANDataset (images listed in data_list/*.txt)"]
    B --> D["TrainerDifIR.prepare_data -> _degrade (clip_len=1)"]
    D --> E["training_step -> GaussianDiffusion.training_losses -> DiTSRModel (4-D input)"]
    F["inference.py --task realsr|faceir"] --> G["sampler.py::Sampler: ImageSpliterTh tiles -> p_sample_loop"]
    G --> H["evaluate.py (no-reference metrics via pyiqa, needs GPU + downloads)"]
```

## 11. What each scenario changes
All rows inherit `configs/vsr_DiT.yaml`; only the listed keys differ. The code path is the same in every scenario.

| Scenario | Changed config keys | Effect on the flow charts |
|---|---|---|
| smoke | tiny model (`model_channels` 32, `swin_depth` 1), `steps: 3`, 20 iterations, `use_amp: False` | same flow, minutes instead of days |
| standard | – | 256 px crops, `num_frames: 5`, 56M parameters |
| small_gpu | `degradation.gt_size: 128`, `num_frames: 3`, smaller model, `attention_resolutions: [32,16,8]` | `resolve_crop` derives `image_size: 32`, patch size 128 |
| single_frame | `num_frames: 1`, `frame_stride: [1,1]` | datasets return `T=1` clips; temporal attention still runs on one frame |
| x2 | `degradation.sf: 2`, `diffusion.params.sf: 2` | `upsample_lq` scales x2; `lq_multiple` = 128 |
| patch inference | `patch_restoration.enabled: True` | `sample_clip` takes the right branch of section 7 and runs section 8 |
| resume | `--resume ckpt` | `resume_from_ckpt` (section 3) restores model, EMA, lr schedule, counters |
| fine-tune | `--set model.ckpt_path=...` | `TrainerBase.build_model` loads the weights into a new run |

## 12. File hierarchy

```
.
├── main.py                       training entrypoint -> trainer.target from the config
├── inference_video.py            video inference CLI -> VideoSampler
├── evaluate_video.py             video metrics CLI
├── inference.py / evaluate.py    upstream image inference / no-reference metrics
├── trainer_video.py              TrainerDifVSR   (video data prep, validation, logging)   extends:
├── trainer.py                    TrainerBase -> TrainerDifIR (degradation, training_step, EMA) [+ image variants]
├── sampler_video.py              VideoSampler    (windows x tiles, blending)              extends:
├── sampler.py                    BaseSampler (loads model + diffusion) -> Sampler (images)
├── patch_restoration.py          PatchAggregatedModel, PatchDiffusiveRestoration, wrap_model
├── configs/
│   ├── vsr_DiT.yaml              base video config (all keys documented)
│   ├── examples/                 scenario configs (use _base_)
│   └── realsr_*.yaml, faceir_*   upstream image configs
├── datapipe/
│   ├── datasets.py               create_dataset(type -> class)
│   └── video_datasets.py         VideoReader, find_videos, VideoRealESRGANDataset, VideoPairedData
├── models/
│   ├── unet.py                   DiTSRModel (5-D input, pixel-unshuffle)
│   ├── swin_transformer.py       BasicLayer, SwinTransformerBlock_AdaLNZero, WindowAttention, TemporalAttention
│   ├── basic_ops.py              frames_to_batch / batch_to_frames, timestep_embedding, norm, conv helpers
│   ├── gaussian_diffusion.py     GaussianDiffusion: q_sample, training_losses, p_sample_loop, upsample_lq
│   ├── script_util.py            create_gaussian_diffusion (config -> diffusion object)
│   └── respace.py, losses.py, resample.py, solvers.py, fp16_util.py, fan.py   support code
├── utils/
│   ├── util_config.py            load_config (_base_, --set)
│   ├── util_crop.py              resolve_crop, check_hr_size, lq_multiple
│   └── util_common.py, util_image.py, util_net.py, util_opts.py, util_sisr.py
├── scripts/                      run_scenario.sh, quickstart.sh, make_demo_data.py
├── basicsr/                      vendored BasicSR pieces (degradations, JPEG, filters, datasets)
├── ldm/                          leftover helper code (util.py, EMA); the autoencoder was removed
└── docs/ARCHITECTURE.md          this file
```

## 13. Tensor layouts cheat sheet

| Where | Layout | Range |
|---|---|---|
| Dataset output `gt` | `[T,3,H,W]` (batch: `[B,T,3,H,W]`) | 0..1 |
| After `prepare_data` / model interface | `[B,3,T,H,W]`; LQ `[B,3,T,H/sf,W/sf]` | -1..1 |
| Inside the model | `[B*T,C,h,w]` (frames folded into the batch) | – |
| After pixel-unshuffle | `[B*T,3p^2,H/p,W/p]`, `p = model.patch_size` | – |
| Diffusion state `x_t`, `x_0` | `[B,3,T,H,W]` (pixel space) | about -1..1 |
| Evaluation | Y channel, 8-bit, per frame | 16..235 |
