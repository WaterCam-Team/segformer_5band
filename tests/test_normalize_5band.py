"""Contract tests for Normalize_5band.

This transform decides checkpoint compatibility: a model fed a distribution it
never saw in training raises no error, it just returns a quietly wrong mask. So
the behaviour of each method, and the metadata the transform publishes, are
pinned here.

Needs the real mmcv/mmseg stack, so it skips where that is not installed (CI
runs the static checks in tools/check_configs.py instead). Run it on a machine
with the inference environment:

    ~/miniforge3/envs/5band/bin/python -m pytest tests/test_normalize_5band.py -v
"""
import numpy as np
import pytest

pytest.importorskip("mmcv", reason="needs the mmcv/mmseg inference stack")

from mmseg.datasets.pipelines.transforms import Normalize_5band  # noqa: E402

FIVE = [10.0, 20.0, 30.0, 40.0, 50.0]


def scene(bands=5, h=4, w=6):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(h, w, bands)).astype(np.uint8)
    # guarantee a non-degenerate range in every band
    img[0, 0, :] = 0
    img[0, 1, :] = 255
    return {"img": img}


def test_minmax_scales_every_band_into_unit_range():
    out = Normalize_5band(mean=FIVE, std=FIVE, method="minmax")(scene())
    img = out["img"]
    assert img.dtype == np.float32
    for b in range(img.shape[2]):
        assert img[:, :, b].min() == pytest.approx(0.0, abs=1e-6)
        assert img[:, :, b].max() == pytest.approx(1.0, abs=1e-3)


def test_minmax_is_per_image_not_global():
    """Two scenes with different ranges both normalise to [0,1] -- which is the
    property that destroys absolute radiometry, and the reason meanstd exists."""
    dim = scene()
    dim["img"] = (dim["img"] // 4).astype(np.uint8)
    out = Normalize_5band(mean=FIVE, std=FIVE, method="minmax")(dim)
    assert out["img"][:, :, 4].max() == pytest.approx(1.0, abs=1e-3)


def test_meanstd_applies_per_band_statistics():
    s = scene()
    raw = s["img"].astype(np.float32).copy()
    out = Normalize_5band(mean=FIVE, std=FIVE, method="meanstd")(s)
    for b in range(5):
        expected = (raw[:, :, b] - FIVE[b]) / (FIVE[b] + 1e-7)
        assert np.allclose(out["img"][:, :, b], expected, atol=1e-4)


def test_meanstd_rejects_a_statistics_count_that_does_not_match_the_bands():
    """Three ImageNet means against five bands used to normalise RGB and leave
    thermal and NIR at raw 0-255, silently. It must raise instead."""
    t = Normalize_5band(mean=[1.0, 2.0, 3.0], std=[1.0, 2.0, 3.0], method="meanstd")
    with pytest.raises(ValueError, match="one mean and std per band"):
        t(scene())


def test_unknown_method_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown method"):
        Normalize_5band(mean=FIVE, std=FIVE, method="zscore")


def test_img_norm_cfg_carries_only_what_tensor2imgs_accepts():
    """Regression guard. mmseg/apis/test.py splats img_norm_cfg straight into
    mmcv.image.tensor2imgs(), which takes mean/std/to_rgb and nothing else, so
    an extra key there breaks every --show / --out-dir run."""
    out = Normalize_5band(mean=FIVE, std=FIVE, method="minmax")(scene())
    assert set(out["img_norm_cfg"]) == {"mean", "std", "to_rgb"}
    assert out["img_norm_method"] == "minmax"


def test_one_band_input_is_handled():
    out = Normalize_5band(mean=[10.0], std=[10.0], method="meanstd")(scene(bands=1))
    assert out["img"].shape[2] == 1
