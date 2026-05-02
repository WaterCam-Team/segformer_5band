# Segmentation Performance — Findings and Improvement Guide

Audit of the `segformer.b0.512x512.flood_5band_2cls.65k` config and supporting
pipeline code. Documents what was found, why it matters, and what to do about it.

---

## Issues Found in the Current Config

### 1. Normalization is broken for 5-band input

**File:** `mmseg/datasets/pipelines/transforms.py` — `Normalize_5band.__call__`

The original `mmcv.imnormalize` call is commented out and replaced with
per-image per-band min-max normalization:

```python
# What the code actually does:
normalized_image[:, :, i] = (band - min_val) / (max_val - min_val + 1e-7)
```

Problems:
- **Destroys absolute radiometric values.** Water has a characteristic low
  reflectance in NIR and SWIR bands. Min-max normalization within each image
  erases that absolute signature — the model cannot learn it.
- **Vulnerable to outlier pixels.** A single cloud, specular reflection, or
  sensor artifact pixel becomes the `max_val` and compresses everything else
  toward zero.
- **The `img_norm_cfg` values in the config are never applied.** They are only
  stored as metadata. The three ImageNet RGB values (`mean=[123.675, 116.28,
  103.53]`) would also be wrong for a 5-band image — 5 values are required.

**Fix:** Compute per-band statistics (mean and std, or 2nd/98th percentile) over
your training dataset, then apply them as a fixed normalization:

```python
# In transforms.py — replace the Normalize_5band __call__ body with:
def __call__(self, results):
    img = results['img'].astype(np.float32)
    for i in range(img.shape[2]):
        img[:, :, i] = (img[:, :, i] - self.mean[i]) / (self.std[i] + 1e-7)
    results['img'] = img
    results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
    return results
```

And update the config to use 5-band statistics:

```python
# In the config — replace the 3-value RGB placeholders with real 5-band stats
# computed from your training set (example values; measure from your data):
img_norm_cfg = dict(
    mean=[B1_mean, B2_mean, B3_mean, B4_mean, B5_mean],
    std=[B1_std,  B2_std,  B3_std,  B4_std,  B5_std],
    to_rgb=False)
```

To compute dataset statistics:
```python
import rasterio, numpy as np, glob

files = glob.glob('/your/data/img_dir/**/*.tif', recursive=True)
sums = np.zeros(5); sq_sums = np.zeros(5); count = 0
for f in files:
    with rasterio.open(f) as src:
        img = src.read().astype(np.float64)  # (5, H, W)
        sums += img.sum(axis=(1, 2))
        sq_sums += (img ** 2).sum(axis=(1, 2))
        count += img.shape[1] * img.shape[2]
means = sums / count
stds = np.sqrt(sq_sums / count - means ** 2)
print('mean:', means.tolist())
print('std:', stds.tolist())
```

---

### 2. Loss function does not account for class imbalance

**Config line:**
```python
loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
```

Plain cross-entropy treats every pixel equally. In flood segmentation, flood
pixels are a small fraction of the scene — the model can achieve low loss by
predicting "no flood" everywhere.

**Lovász loss is already implemented** in this repo at
`mmseg/models/losses/lovasz_loss.py`. It directly optimizes the Jaccard index
(IoU), making it ideal for class-imbalanced binary segmentation.

**Fix:** Switch to Lovász or add a combined loss:

```python
# Option A — Lovász alone (directly optimizes IoU):
loss_decode=dict(type='LovaszLoss', loss_type='binary', reduction='none')

# Option B — weighted cross-entropy (simpler, good baseline):
loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                 class_weight=[0.1, 0.9])  # [background, flood] — tune to your ratio

# Option C — combined CE + Lovász:
loss_decode=[
    dict(type='CrossEntropyLoss', loss_weight=0.5,
         class_weight=[0.1, 0.9]),
    dict(type='LovaszLoss', loss_type='binary', loss_weight=0.5,
         reduction='none'),
]
```

---

### 3. Training iterations set to 100 (debug value)

```python
runner = dict(type='IterBasedRunner', max_iters=100)  # should be 65000
```

The filename says `65k` but `max_iters=100`. The warmup is set to 1500 iters —
longer than the whole training run, meaning the LR never actually ramps up.

**Fix:**
```python
runner = dict(type='IterBasedRunner', max_iters=65000)
evaluation = dict(interval=4000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=4000)
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,   # correct once max_iters=65000
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
```

---

### 4. Test-time augmentation (TTA) disabled

```python
# img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],   # commented out
flip=False,
```

TTA averages predictions over multiple scales and flips. It is free accuracy
at inference time with no retraining.

**Fix:** Enable for evaluation runs (not needed for production speed-sensitive inference):
```python
img_ratios=[0.75, 1.0, 1.25],
flip=True,
```

---

### 5. Data augmentation is minimal

`PhotoMetricDistortion` is commented out. For satellite imagery, radiometric
variation between scenes (atmospheric haze, sensor gain, illumination angle) is
significant. Adding it improves generalization.

**Fix:** Uncomment and add rotation:
```python
dict(type='PhotoMetricDistortion'),
dict(type='RandomRotate', prob=0.5, degree=(-15, 15)),
```

---

## Model Capacity

The current model is **SegFormer-B0** (3.7M parameters), the smallest variant.
Larger variants retain the same architecture but use wider/deeper backbones:

| Model | Params | Typical flood IoU gain vs B0 |
|-------|--------|-------------------------------|
| B0    | 3.7M   | baseline                      |
| B2    | 25M    | +3–5 IoU                      |
| B4    | 65M    | +5–8 IoU                      |
| B5    | 82M    | marginal over B4              |

B2 configs exist in `local_configs/segformer/B2/`. Upgrade by changing the
`pretrained` and `backbone` entries, updating `in_channels` to match B2
(`[64, 128, 320, 512]`), and downloading `mit_b2.pth` from the
[SegFormer releases](https://github.com/NVlabs/SegFormer/releases).

---

## Summary — Priority Order

| Priority | Change | Expected gain | Effort |
|----------|--------|---------------|--------|
| 1 (critical) | Fix `Normalize_5band` + compute real 5-band stats | Large — model may not be learning band signatures at all | Medium |
| 2 (critical) | Set `max_iters=65000` | Large — model is currently massively undertrained | Low |
| 3 (high) | Switch to Lovász or weighted CE loss | Medium — directly addresses flood/background imbalance | Low |
| 4 (medium) | Upgrade B0 → B2 backbone | Medium — +3–5 IoU at cost of slower inference | Low |
| 5 (medium) | Enable TTA at inference | Small–Medium — free accuracy, no retraining | Low |
| 6 (low) | Enable `PhotoMetricDistortion` + rotation augmentation | Small — better scene generalization | Low |
| 7 (low) | Post-processing: morphological closing on flood mask | Small — cleans isolated false-positive pixels | Low |
