"""Model-level check: synthetic 5-band TIFF -> ONNX export -> onnxruntime.

A tiny randomly-initialised network stands in for SegFormer so this runs in
seconds and needs no checkpoint; what it pins is the path around the model:
dynamic H/W, output shape, and that the schema-1 metadata survives the export
(`metadata_props` read back by onnxruntime, `model.json` sidecar with the sha).
Skips where torch / onnx / onnxruntime / rasterio are not installed. No real
captures are used: the TIFF is generated here.
"""
import json
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')
rasterio = pytest.importorskip('rasterio')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
import export_onnx_5band as E  # noqa: E402
import model_metadata as MM  # noqa: E402

BANDS = ['red', 'green', 'blue', 'thermal', 'nir']
H, W = 64, 96


class Tiny(torch.nn.Module):
    """5 -> 2 channels, stride-1; the same input/output contract as the real graph."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(5, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def tiff(tmp_path):
    path = tmp_path / 'synthetic_5_band.tiff'
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(5, H, W), dtype=np.uint8)
    with rasterio.open(path, 'w', driver='GTiff', height=H, width=W, count=5,
                       dtype='uint8') as dst:
        dst.write(data)
        dst.update_tags(BAND_ORDER=','.join(BANDS))
    return path


class Cfg:
    """Just enough of an mmcv Config for metadata_from_config."""

    class data:
        test = type('T', (), {'pipeline': [dict(
            type='MultiScaleFlipAug', transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='Normalize_5band', method='minmax',
                     mean=[0] * 3, std=[1] * 3)])]})


def test_export_to_onnxruntime_keeps_shape_and_metadata(tmp_path, tiff):
    out = str(tmp_path / 'tiny.onnx')
    props = E.metadata_from_config(
        Cfg, model_name='tiny', model_version='test', bands=BANDS,
        classes=['background', 'water'], height=H, width=W)
    assert props['normalization'] == 'minmax'
    assert props['resize_interpolation'] == 'bilinear'

    model = Tiny().eval()
    E.export_graph(model, torch.randn(1, 5, H, W), out, opset=12)
    MM.stamp_onnx(out, props)
    sidecar = MM.write_sidecar(out, props)

    with rasterio.open(tiff) as src:
        img = src.read().astype(np.float32)
    lo = img.min(axis=(1, 2), keepdims=True)
    hi = img.max(axis=(1, 2), keepdims=True)
    x = ((img - lo) / (hi - lo))[None]

    sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    got = sess.run(None, {'input': x})[0]
    assert got.shape == (1, 2, H, W)
    # a size the graph was not traced at (dynamic H/W)
    other = sess.run(None, {'input': np.zeros((1, 5, 32, 64), np.float32)})[0]
    assert other.shape == (1, 2, 32, 64)
    with torch.no_grad():
        assert np.allclose(got, model(torch.from_numpy(x)).numpy(), atol=1e-4)

    meta = sess.get_modelmeta().custom_metadata_map
    assert dict(meta) == props
    side = json.load(open(sidecar))
    assert side['sha256'] == MM.sha256_file(out)
    assert side['bands'] == BANDS and side['in_channels'] == 5
