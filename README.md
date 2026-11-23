# segformer_5band

Water segmentation for the HazMapper sensor nodes, built on SegFormer. The
repository is a fork of the SegFormer code (mmsegmentation based) with 5-band
data loading, training configs, and an ONNX export for the nodes.

## The model

- Architecture: SegFormer with the MiT-B0 backbone (the deployed model). B1 to
  B5 configs exist under `local_configs/segformer/`; the modality comparison
  uses B2.
- Classes: 2, `background` (0) and `water` (1).
- Input: 5 bands, in this order: red, green, blue, thermal (FLIR Lepton, LWIR),
  NIR difference. The backbone is built with `in_chans=5`.
- Normalisation is a declared config value (`minmax` or `meanstd`); see
  `NEXT_CHECKPOINT.md` section 1.
- Every exported `.onnx` carries its bands, classes and normalisation as
  metadata; see `docs/MODEL_METADATA.md`.

## Main files

| Path | What |
|---|---|
| `local_configs/segformer/B0/segformer.b0.512x512.flood_5band_2cls.65k.py` | training config of the deployed 5-band B0 model |
| `local_configs/segformer/B0/segformer.b0.512x512.flood_5band_2cls.65k.test.py` | matching test and export config |
| `local_configs/segformer/B2/segformer.b2.512x512.flood_{5band,rgb_std,rgb_nofilt,lwir}_2cls.65k.py` | B2 configs for the modality comparison |
| `local_configs/_base_/models/segformer.py` | base model definition |
| `mmseg/datasets/dataset_5band.py`, `custom_5band.py` | 5-band dataset (`.tif` images, `.png` masks) |
| `tools/train.py`, `tools/test.py`, `tools/test_no_label.py` | train and test |
| `tools/export_onnx_5band.py` | ONNX export with metadata |
| `segment_tiff_5band.py` | run an exported model on a 5-band TIFF |

Config names follow `segformer.<backbone>.<crop size>.<dataset>.<iterations>`.
A `.test.py` suffix marks the variant used for testing and export. The
`configs/` directory is the unchanged upstream mmsegmentation set; the
SegFormer configs used here are in `local_configs/`.

## Train and test

Train:

`python ./tools/train.py`

Test without ground truth:

`python ./tools/test_no_label.py`

`tools/test_no_label.py` defaults to the 5-band B0 test config and the `_huantao` `iter_100.pth` checkpoint; override with `--config` and `--checkpoint`.

## Export to ONNX

See [docs/ONNX_DEPLOYMENT.md](docs/ONNX_DEPLOYMENT.md). In short:

```bash
PYTHONPATH=. python tools/export_onnx_5band.py \
    --config local_configs/segformer/B0/segformer.b0.512x512.flood_5band_2cls.65k.test.py \
    --checkpoint work_dirs/segformer.b0.512x512.flood_5band_2cls.65k_huantao/iter_100.pth \
    --out segformer_5band_dyn.onnx
```

## Checkpoints and models

| File | What |
|---|---|
| `work_dirs/segformer.b0.512x512.flood_5band_2cls.65k_huantao/iter_100.pth` | the 5-band B0 checkpoint (100 training iterations); the one exported for the nodes |
| `work_dirs/segformer.b0.512x512.flood_5band_2cls.65k/iter_100.pth` | a second copy of the 5-band run |
| `segformer_5band_dyn.onnx` | the exported 5-band model the nodes serve |
| `segformer_5band_fp32.onnx` | symlink to the file above (the name the node daemon's FP32 fallback loads) |
| `pretrained/mit_b0.pth`, `checkpoints/segformer.b0.512x512.ade.160k.pth` | pretrained starting weights |

Which model runs where:

- The 5-band `iter_100` checkpoint is the default on the nodes.
- An RGB model (called `type_pool`) is used as the fallback when a unit's NIR
  (IR-cut) filter is unreliable, because the NIR band is then not trustworthy.
  It is trained in the photo-masks repository, not here, and is not part of
  this repository.

The `iter_100` checkpoint is barely trained and its accuracy has not been
measured; see `docs/ONNX_DEPLOYMENT.md` (Limits) and `NEXT_CHECKPOINT.md`.

## Environment

`5band_environment.yml` and `requirements.txt` describe the environments.
`5band_environment.yml` pins the older stack used on the nodes (Python 3.9,
torch 1.7.1, mmcv-full 1.3.0, onnxruntime 1.19.2). Running an exported model
needs only `onnxruntime`. See `docs/ONNX_DEPLOYMENT.md`.

## Docs map

| Doc | What |
|---|---|
| `docs/ONNX_DEPLOYMENT.md` | ONNX export, accuracy of the conversion, padding caveat, the committed model |
| `docs/MODEL_METADATA.md` | the metadata schema every exported model carries (shared with photo-masks) |
| `NEXT_CHECKPOINT.md` | what the next checkpoint must fix: normalisation, padding, band order, 5-band versus 3-band |
| `PERFORMANCE.md` | audit of the config and pipeline, with a status per issue |
| `MODALITY_COMPARISON.md` | the RGB, RGB-without-filter, LWIR and 5-band comparison with B2 models |

## License

NVIDIA Source Code License (non-commercial use); see `LICENSE`.
