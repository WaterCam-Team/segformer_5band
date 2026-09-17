# Running the 5-band model through ONNX Runtime

The deployment target is a Raspberry Pi 4B (Cortex-A72, 4 cores, 3.7 GiB)
running the `5band` conda environment: Python 3.9, torch 1.7.1, mmcv 1.3.0.
That torch was built for aarch64 in 2020, before most of PyTorch's ARM
optimisation work, and it is the single largest cost in inference.

ONNX Runtime 1.19.2 runs the same graph **19.8× faster** with numerically
identical output.

## Usage

```bash
# one-off export
PYTHONPATH=. python tools/export_onnx_5band.py \
    --config local_configs/segformer/B0/segformer.b0.512x512.flood_5band_2cls.65k.test.py \
    --checkpoint work_dirs/segformer.b0.512x512.flood_5band_2cls.65k_huantao/iter_100.pth \
    --out segformer_5band_dyn.onnx

# inference
python segment_tiff_5band.py input.tiff --onnx segformer_5band_dyn.onnx
```

`onnxruntime` is the only new dependency; torch is still needed to export, but
not to run.

## Results

Measured on 64 real 5-band captures (Onondaga Lake Park, 1296×972, uint8).

| | per image |
|---|---|
| torch 1.7.1 | 29.7 s |
| ONNX Runtime 1.19.2 | **1.50 s** |
| wall clock incl. startup and IO | 35 s → 11 s |

**The conversion is exact.** Against torch on identical input, across all 64:

| | |
|---|---|
| argmax agreement | mean **99.9999%**, worst **99.9994%** |
| max absolute logit difference | ~5e-06 |

Verified at the traced resolution (512×1024) and at an untraced 384×640, so the
dynamic axes genuinely work rather than the graph being secretly static.

## How the export works

One thing pins the graph to its traced resolution: `SegFormerHead` resizes
`_c4/_c3/_c2` to `size=c1.size()[2:]` (`segformer_head.py:73,76,79`), and
`mmseg.ops.resize` converts a `torch.Size` into Python ints.

Those three are always exact power-of-two ratios — strides 32/16/8 against c1's
4 — so `scale_factor` is equivalent **provided input dimensions divide by 32**.
For other sizes ceil division in the strided convolutions makes the two differ
by a pixel, so the head itself is deliberately left alone and
`patch_resize_for_export()` applies the substitution **during tracing only**.

## Padding is not free — read this before trusting a mask

The exported graph needs dimensions divisible by 32. Captures are 1296×972,
which after the pipeline's `keep_ratio` resize becomes 512×683 — so
`_inference_onnx` pads to 512×704, and warns when it does.

That padding changes the result. MiT's spatial-reduction attention pools
globally, so a padded strip shifts predictions across the **whole image**, not
just at the border. Measured torch-against-torch, with no ONNX involved, over
the same 64 captures:

| | |
|---|---|
| pixels agreeing | mean 96.73%, range 91.37–98.79% |
| water fraction shift | mean +0.89 pts, range −3.85 to +4.88 |
| images shifted more than 1 point | **39 of 64** |

The direction is **not** systematic — it moves both ways depending on content —
and no padding mode avoids it; `replicate` and `reflect` simply bias
differently. The magnitude tracks how confident the model is: the converged
40k-iteration type_pool model moves 0.79% of pixels under the same treatment,
this 100-iteration checkpoint 3.3% on average.

**The fix is to resize rather than pad.** Choose an `img_scale` in the test
pipeline that lands on a multiple of 32 and the perturbation disappears
entirely instead of being bounded.

## Limits

- **No accuracy has been measured, at all.** There is no labelled 5-band
  dataset on the Pi or the laptop. Everything above is about whether the
  deployment path preserves the model's output, never whether that output is
  correct.
- **The checkpoint is `iter_100`** — 100 training iterations, in a directory
  named `65k`. Whatever it segments, it is not a trained model.

## If you are about to label data

Two details in `dataset_5band` will silently waste the effort:

| trap | what happens | what to do |
|---|---|---|
| `img_suffix='.tif'` (`dataset_5band.py:23`) but captures are `.tiff` | no images found, empty dataset | rename to `.tif`, or change the suffix |
| `ignore_index=255` (`custom_5band.py:83`) | a `{0, 255}` mask marks every water pixel **ignored**, not water | write masks as `{0, 1}` |

The existing `water_mask.png` in `20250803-0331-1` is `{0, 255}`, so it would
need converting. Expected layout:

```
<root>/img_dir/val/<name>.tif     5-band capture
<root>/ann_dir/val/<name>.png     single channel, 0 = background, 1 = water
```

Matching basenames, `reduce_zero_label=False`, classes `('background', 'water')`.

Once labels exist, the first thing worth measuring is whether the resize-to-32
pipeline change costs anything — it can then be compared properly rather than
argued about.
