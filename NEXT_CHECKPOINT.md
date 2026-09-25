# Preparing the next checkpoint

Written 2026-09-25, after a week of measuring the deployed pipeline on node `ufo-01-01-005`.
Everything here is measured unless it says otherwise.

The current checkpoint is `work_dirs/segformer.b0.512x512.flood_5band_2cls.65k_huantao/iter_100.pth`
— **SegFormer-B0, 100 iterations**. It is barely trained, and several findings below are consequences
of that rather than of the architecture.

---

## 1. Normalisation is the decision that blocks everything else

`Normalize_5band` had shipped in two incompatible forms: per-image min-max on the node, and mean/std
over `range(len(self.mean))` here — which, with three ImageNet means against a five-band model,
normalised R, G and B and left thermal and NIR at raw 0-255. Neither announced itself.

It is now a declared config value (`method='minmax'` or `'meanstd'`), and `meanstd` raises if the
statistics count does not match the band count.

**`iter_100.pth` was trained under min-max, so `minmax` is the default and the node is unchanged.**

**For the next checkpoint, train under `meanstd` with real per-band statistics.** Per-image min-max
normalises each frame independently, which erases the absolute radiometry that makes NIR worth
carrying: water's low NIR reflectance is an *absolute* signature. It is also hostage to one specular
or dead pixel setting a band's range. This is `PERFORMANCE.md`'s original complaint and it still
stands.

Compute the statistics over the *training split only* and freeze them into the checkpoint:

    uv run --project annotator python -m training.stats --modality fiveband

`photo_processing/training/stats.py` does this in one streaming pass with exact 256-bin histograms.
Put the resulting five means and five stds into the config and set `method='meanstd'`.

Do not change normalisation without retraining. Feeding a model a distribution it never saw produces
no error, only a wrong mask — measured elsewhere in this project at 99.68% of a frame called water
against 52.4% correct.

## 2. The NIR band is lossy before the model ever sees it

`coreg/coreg_multiple.py` builds band 4 as `cv2.subtract(red(NIR-ON), red(NIR-OFF))`. Three problems
compound:

- `cv2.subtract` **saturates at 0** on uint8, so every pixel where NIR-OFF >= NIR-ON clips and is
  unrecoverable.
- The two frames are **separate exposures**. `photo_processing/training/modalities.py` measured the
  drift: `band0 + band4` against `red(NIR-ON)` ranges from MAE 0.25 to 28.9 across sessions.
- It is a **temporal** difference, so anything moving between the two captures writes false NIR.

**Proposed for the next capture generation: keep five bands, but make band 4 the raw NIR-ON red
channel** rather than the clipped difference. The old difference stays recoverable inside the network
as `band4 - band0`, because band 0 is exactly red(NIR-OFF), and it is then signed and unclipped. Same
band count, same model shape, no extra storage.

This changes the input distribution, so it lands with the retrain, not before.

Also record per-frame exposure and gain in the TIFF metadata. Today a real NIR change and an
auto-exposure change are indistinguishable.

## 3. Padding shifts predictions, so size inputs to a multiple of 32

Measured here over 64 real captures (`segment_tiff_5band.py`, `_inference_onnx`): padding 512x683 to
512x704 moves **3.27% of pixels on average** (agreement 96.73%, range 91.37-98.79%), with water
fraction shifting by more than a point in 39 of 64 images. MiT's spatial-reduction attention pools
globally, so a padded strip perturbs predictions across the whole frame, not just at the border. The
direction is not systematic.

The magnitude tracks model confidence: the 40k-iteration type_pool model moves 0.79% of pixels,
`iter_100` moves 3.3%. **A better-trained checkpoint should be markedly less sensitive**, which is
worth re-measuring once it exists.

The clean fix is to choose a keep-ratio inference size that is already a multiple of 32 rather than
padding to one. Prepared on the `awaiting-new-checkpoint` branch in SU-WaterCam, because it changes
outputs and cannot be validated against `iter_100` meaningfully.

## 4. Registration is stochastic, and the transform should be frozen

Mattes mutual information samples the images, so the same scene solved repeatedly gives different
answers. Measured on node 005: three solves of one scene returned tx of -33.22, -29.96 and -22.65,
and across all observations that scene spanned about 101 px.

The transform cache was writing but never being read back — it compared the concrete SimpleITK class
against the configured `TRANSFORM_TYPE`, and registration returns a `CompositeTransform` even when
the config says `affine`, so every entry was rejected. Fixed in SU-WaterCam. Co-registration drops
from about 30 s per capture to about 1 s once the cache is warm.

**Caching removes the variance, not the bias.** Seed the cache deliberately with
`tools/seed_registration_cache.py`, which solves N times and freezes the medoid. **Seed it from a
high-contrast field capture, not from node 005**: 005 is indoors and thermally flat, and four solves
there disagreed by 42 px at the median, which the tool warns about.

Also note there is no `camera_calibration.json` on the node, so lens undistortion never runs and the
transform is fitted between two distorted images.

## 4b. The five-band stem is not warm-started by this training path

Found while checking a review comment. Building the B0 five-band config with `pretrained=
'pretrained/mit_b0.pth'` **does not fail** — mmcv's `load_checkpoint(strict=False)` logs the
mismatch and skips that tensor:

    size mismatch for patch_embed1.proj.weight: copying a param with shape
    torch.Size([32, 3, 7, 7]) from checkpoint, the shape in current model is
    torch.Size([32, 5, 7, 7])

The build succeeds and the stem is correctly `(32, 5, 7, 7)`. But **it is randomly initialised**,
because the only tensor that could have seeded it was the one skipped. Everything downstream of the
stem is warm-started; the first convolution is not.

`photo_processing/training/models/segformer.py` does this properly: it rebuilds the stem and
warm-starts it from the RGB filters (`base.SegModel.adapt_stem`). **This training path has no
equivalent.** At 100 iterations it hardly matters, but for a real training run a randomly
initialised first layer against warm-started everything else is worth fixing — seed the RGB
channels from the pretrained filters and the thermal and NIR channels from their mean.

## 5. What the training data situation actually is

- **Masks are binary, 2-class**, painted on the 1296x972 grid. Three class decisions are still open
  in `annotator/LABELING_GUIDE.md`: ice and snow, wet pavement, thin sheet flow. **Settle these before
  mass labelling.** At this data scale, label inconsistency costs more than any architecture choice,
  and relabelling is the most expensive mistake available.
- **Split by capture session, not randomly.** `photo_processing/training/cv.py` does leave-one-session-out
  and documents why: the rig's repeated captures put near-duplicate frames on both sides of a random
  split, so a random split leaks and inflates scores.
- **Report water IoU, not mIoU.** Background dominates.
- Keep photometric augmentation restricted to the optical channels, as `modalities.py` already
  encodes. Brightness jitter on a thermal or NIR band fakes radiometry the sensor cannot produce.
- The ablations are already defined (`rgb_std`, `rgb_nir`, `rgb_thermal`, `fiveband`). **There is
  currently no evidence that five bands beat three.** That comparison is the result worth having.

## 6. Checklist

1. Settle the three open label decisions.
2. Label, using the annotator; keep session grouping intact.
3. Decide band 4: keep the clipped difference, or move to raw NIR-ON red.
4. Run `training.stats` over the training split; put five means and five stds in the config.
5. Set `method='meanstd'`. Train.
6. Re-measure padding sensitivity against the new checkpoint; adopt /32 sizing if it still matters.
7. Re-seed the registration transform from a high-contrast field scene.
8. Re-export ONNX through the metadata-aware exporter so the graph declares its normalisation, and
   re-run the node benchmarks.
