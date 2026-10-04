# Modality Comparison Pipeline

Compares water segmentation accuracy across four imaging modalities using
matched SegFormer-B2 models trained on the same co-registered scenes.

## Modalities

| Name | Input | Channels | Source file (from coreg) |
|------|-------|----------|--------------------------|
| `rgb_std` | Standard RGB | 3 | `*-NIR-OFF.jpg` |
| `rgb_nofilt` | RGB without NIR filter | 3 | `*-NIR-ON.jpg` |
| `lwir` | FLIR Lepton thermal | 1 | `*.pgm` |
| `fiveband` | R, G, B, Thermal, NIR | 5 | `color_preserved_5_band.tiff` |

NIR band = `red(NIR-ON) − red(NIR-OFF)` (computed during co-registration).
All four views of each scene share one ground-truth water mask.

---

## Repository Layout

```
coreg/
└── water_autolabel.py        # Step 1 — generate candidate masks

segformer_5band/
├── mmseg/
│   ├── datasets/
│   │   ├── dataset_rgb.py    # dataset_rgb_std, dataset_rgb_nofilt
│   │   ├── dataset_lwir.py   # dataset_lwir
│   │   └── pipelines/
│   │       ├── loading.py    # Load_LWIR_ImageFromFile
│   │       └── transforms.py # Normalize_5band (fixed), Normalize_1band
│   └── models/backbones/
│       └── mix_transformer.py  # in_chans now configurable for all mit_bX
├── local_configs/segformer/B2/
│   ├── segformer.b2.512x512.flood_rgb_std_2cls.65k.py
│   ├── segformer.b2.512x512.flood_rgb_nofilt_2cls.65k.py
│   ├── segformer.b2.512x512.flood_lwir_2cls.65k.py
│   └── segformer.b2.512x512.flood_5band_2cls.65k.py
└── tools/
    ├── split_dataset.py      # Step 3 — organise scenes into dataset
    ├── compute_band_stats.py # Step 4 — compute normalisation values
    ├── train.py              # Step 6 — train (existing)
    └── compare_modalities.py # Step 7 — evaluation comparison table
```

---

## End-to-End Workflow

### Step 1 — Auto-label candidate masks

```bash
python ../coreg/water_autolabel.py /path/to/scenes --preview
```

Reads `color_preserved_5_band.tiff` from each scene directory. Thresholds
band 5 (NIR — water absorbs NIR, producing low values) to produce
`water_mask_auto.png`. The `--preview` flag writes a side-by-side
false-colour overlay JPEG for rapid visual QA.

Default threshold `--threshold 0.25` (relative, 0–1). Tune if too many
false positives (raise) or false negatives (lower).

### Step 2 — Manual review and correction

Open `false_color_composite.jpg` (NIR=red, Thermal=green — water visible)
alongside `water_mask_auto.png` in **LabelMe** or **CVAT**. Correct
misclassified regions. Save the corrected mask as `water_mask.png` in
the same directory.

Label format: single-channel PNG, 0 = background, 1 = water.

### Step 3 — Organise into dataset

```bash
python tools/split_dataset.py /path/to/scenes data/modality_comparison \
    --train 0.70 --val 0.15 --test 0.15 --seed 42
```

Finds all scene directories containing `water_mask.png` plus all four
modality files. Copies files into a per-modality directory structure:

```
data/modality_comparison/
  rgb_std/img_dir/{train,val,test}/
  rgb_std/ann_dir/{train,val,test}/
  rgb_nofilt/...
  lwir/...
  fiveband/...
```

### Step 4 — Compute normalisation statistics

Run once per modality against the training split:

```bash
for MODALITY in rgb_std rgb_nofilt lwir fiveband; do
    python tools/compute_band_stats.py \
        --data_root data/modality_comparison \
        --modality $MODALITY
done
```

Each run prints a paste-ready `img_norm_cfg` snippet. Replace the
placeholder `mean=[0.0,...]  std=[1.0,...]` values in the corresponding
`local_configs/segformer/B2/` config file.

### Step 5 — Download pretrained weights

```bash
mkdir -p pretrained
# mit_b2 (used by rgb_std, rgb_nofilt, fiveband configs):
wget -P pretrained https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth
mv pretrained/mit_b2_20220624-66e8bf70.pth pretrained/mit_b2.pth
```

> **LWIR config** (`flood_lwir_2cls.65k.py`) sets `pretrained=None` because
> the 3-channel pretrained `patch_embed1` weight cannot be loaded into a
> 1-channel model without adaptation. The LWIR model trains from random
> init. If you want to adapt the pretrained weights, average the 3 input
> channels of `patch_embed1.proj.weight` to produce a 1-channel weight
> and load it manually before training.

### Step 6 — Train all four models

```bash
for CONFIG in \
    local_configs/segformer/B2/segformer.b2.512x512.flood_rgb_std_2cls.65k.py \
    local_configs/segformer/B2/segformer.b2.512x512.flood_rgb_nofilt_2cls.65k.py \
    local_configs/segformer/B2/segformer.b2.512x512.flood_lwir_2cls.65k.py \
    local_configs/segformer/B2/segformer.b2.512x512.flood_5band_2cls.65k.py; do
    python tools/train.py $CONFIG --work-dir work_dirs/$(basename $CONFIG .py)
done
```

### Step 7 — Compare results

```bash
python tools/compare_modalities.py \
    --data_root data/modality_comparison \
    --configs \
        local_configs/segformer/B2/segformer.b2.512x512.flood_rgb_std_2cls.65k.py \
        local_configs/segformer/B2/segformer.b2.512x512.flood_rgb_nofilt_2cls.65k.py \
        local_configs/segformer/B2/segformer.b2.512x512.flood_lwir_2cls.65k.py \
        local_configs/segformer/B2/segformer.b2.512x512.flood_5band_2cls.65k.py \
    --checkpoints \
        work_dirs/segformer.b2.512x512.flood_rgb_std_2cls.65k/latest.pth \
        work_dirs/segformer.b2.512x512.flood_rgb_nofilt_2cls.65k/latest.pth \
        work_dirs/segformer.b2.512x512.flood_lwir_2cls.65k/latest.pth \
        work_dirs/segformer.b2.512x512.flood_5band_2cls.65k/latest.pth
```

Prints a table of mIoU, water IoU, F1, precision, and recall per modality.
Saves results to `work_dirs/modality_comparison.csv`.

---

## Config Design Decisions

All four B2 configs share the same design choices for a fair comparison:

| Setting | Value | Reason |
|---------|-------|--------|
| Backbone | SegFormer-B2 | Balance of accuracy and inference speed |
| Loss | CE (w=0.5, class_weight=[0.3,0.7]) + Lovász (w=0.5) | Handles class imbalance; Lovász directly optimises IoU |
| max_iters | 65 000 | Full training schedule per filename |
| TTA | scales [0.75, 1.0, 1.25] + flip | Free accuracy at evaluation |
| Augmentation | RandomFlip + PhotoMetricDistortion | Generalization across lighting/sensor variation |
| crop_size | 512×512 | GPU memory; adjust for larger GPU |
| samples_per_gpu | 4 (3ch/1ch), 2 (5band) | 5-band uses more memory |

---

## Notes on the LWIR Modality

- FLIR Lepton native resolution is 160×120. During co-registration
  (`coreg_multiple.py`) it is upsampled to match the optical image size.
  The resulting thermal band is spatially smooth.
- Water temperature varies with season, time of day, and weather.
  The LWIR model is expected to underperform in conditions where water
  and surrounding land are at similar temperatures (e.g. dawn, rain).
- No pretrained weights are used; the model trains from scratch.
  More training data or longer `max_iters` may be needed to compensate.
