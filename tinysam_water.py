#!/usr/bin/env python3
"""
Water segmentation using TinySAM with NDWI-guided point prompt.

Uses the NIR band from a 5-band TIFF to compute NDWI and find a reliable
water point, then prompts TinySAM to segment the connected water body.

Usage:
    TINYSAM_PATH=/path/to/TinySAM python tinysam_water.py <scene_dir> [--multi-point N] [--output output.png]

TINYSAM_PATH is a checkout of TinySAM; TINYSAM_WEIGHTS defaults to
$TINYSAM_PATH/weights/tinysam.pth.

Expects in scene_dir:
    color_preserved_5_band.tiff  (bands: R, G, B, Thermal, NIR)
    *-NIR-OFF.jpg                (standard RGB for SAM input)
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import cv2
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Where the TinySAM repo and its weights live (not vendored here).
TINYSAM_PATH = os.environ.get('TINYSAM_PATH', '')
TINYSAM_WEIGHTS = os.environ.get(
    'TINYSAM_WEIGHTS', os.path.join(TINYSAM_PATH, 'weights', 'tinysam.pth'))

if TINYSAM_PATH:
    sys.path.append(TINYSAM_PATH)


def load_5band(tiff_path):
    with rasterio.open(tiff_path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)
    return data


def pick_water_points_from_bands(bands, n=5):
    """
    Detect water using NIR + thermal:
      - Water:      low NIR  (absorbed) AND cool thermal
      - Rocks:      low NIR             AND warm thermal  <- excluded by thermal cutoff
      - Vegetation: high NIR            AND cool thermal  <- excluded by NIR cutoff

    Band layout: R=0, G=1, B=2, Thermal=3, NIR_diff=4
    """
    nir     = bands[4].astype(np.float32)
    thermal = bands[3].astype(np.float32)

    # Per-image adaptive thresholds
    # Water: low NIR (absorbed) AND cool thermal — excludes hot rocks (high thermal)
    #        and vegetation (high NIR). No luminance filter: gray is unreliable.
    nir_cutoff     = np.percentile(nir,     50)
    thermal_cutoff = np.percentile(thermal, 50)

    water_mask = (nir < nir_cutoff) & (thermal < thermal_cutoff)

    # Exclude a 3% border margin to avoid edge/sky strip artifacts
    h, w = water_mask.shape
    margin = max(5, int(min(h, w) * 0.03))
    water_mask[:margin, :]  = 0
    water_mask[-margin:, :] = 0
    water_mask[:, :margin]  = 0
    water_mask[:, -margin:] = 0

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    water_mask = cv2.morphologyEx(water_mask.astype(np.uint8), cv2.MORPH_OPEN,  kernel, iterations=2)
    water_mask = cv2.morphologyEx(water_mask,                  cv2.MORPH_CLOSE, kernel, iterations=2)

    if water_mask.sum() == 0:
        return None, water_mask

    # Pick centroids of largest blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(water_mask)
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    pts = []
    for label in order:
        cx, cy = int(centroids[label][0]), int(centroids[label][1])
        if water_mask[cy, cx]:
            pts.append([cx, cy])
        else:
            ys, xs = np.where(labels == label)
            if len(xs):
                idx = np.argmin((xs - cx)**2 + (ys - cy)**2)
                pts.append([xs[idx], ys[idx]])
        if len(pts) >= n:
            break
    return (np.array(pts) if pts else None), water_mask


def find_rgb_image(scene_dir):
    scene_dir = Path(scene_dir)
    candidates = sorted(scene_dir.glob('*-NIR-OFF.jpg'))
    if candidates:
        return candidates[0]
    # fallback
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        found = sorted(scene_dir.glob(ext))
        if found:
            return found[0]
    return None


def run(scene_dir, n_points, output_path):
    scene_dir = Path(scene_dir)
    tiff = scene_dir / 'color_preserved_5_band.tiff'
    if not tiff.exists():
        tiff = scene_dir / 'final_5_band.tiff'
    if not tiff.exists():
        raise FileNotFoundError(f'No 5-band TIFF in {scene_dir}')

    rgb_path = find_rgb_image(scene_dir)
    if rgb_path is None:
        raise FileNotFoundError(f'No RGB image in {scene_dir}')

    print(f'5-band: {tiff.name}')
    print(f'RGB:    {rgb_path.name}')

    # Load RGB for SAM
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_h, rgb_w = rgb.shape[:2]

    # Pick prompt points using NIR + thermal combined water detection
    bands = load_5band(tiff)
    tiff_h, tiff_w = bands.shape[1], bands.shape[2]

    print('Detecting water pixels using NIR + thermal...')
    water_pts_tiff, spectral_mask = pick_water_points_from_bands(bands, n=n_points)
    source = 'NIR+thermal'

    if water_pts_tiff is None:
        raise RuntimeError('No water candidate pixels found in scene')

    print(f'Prompt points ({source}, TIFF coords x,y): {water_pts_tiff}')

    # Scale points from TIFF coords to RGB coords
    if (tiff_h, tiff_w) != (rgb_h, rgb_w):
        sx = rgb_w / tiff_w
        sy = rgb_h / tiff_h
        water_pts = (water_pts_tiff * np.array([sx, sy])).astype(int)
        print(f'Rescaled to RGB size ({rgb_w}x{rgb_h}): {water_pts}')
    else:
        water_pts = water_pts_tiff

    # Load TinySAM
    if not os.path.isfile(TINYSAM_WEIGHTS):
        sys.exit(f'TinySAM weights not found at {TINYSAM_WEIGHTS!r}; set TINYSAM_PATH '
                 'or TINYSAM_WEIGHTS')
    from tinysam import sam_model_registry, SamPredictor
    print('Loading TinySAM...')
    sam = sam_model_registry['vit_t'](checkpoint=TINYSAM_WEIGHTS)
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)

    # Predict with all water points as positive prompts
    labels = np.ones(len(water_pts), dtype=int)
    masks, scores, _ = predictor.predict(
        point_coords=water_pts,
        point_labels=labels,
    )
    best = masks[scores.argmax()]

    # Save binary mask (0/255)
    mask_path = output_path or str(scene_dir / 'tinysam_water_mask.png')
    cv2.imwrite(mask_path, (best * 255).astype(np.uint8))
    print(f'Binary mask -> {mask_path}')

    # Save overlay visualisation
    overlay = rgb.copy()
    overlay[best] = (overlay[best] * 0.5 + np.array([0, 102, 204]) * 0.5).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].scatter(water_pts[:, 0], water_pts[:, 1], c='lime', s=120, marker='*',
                    edgecolors='black', linewidths=0.5, zorder=5)
    axes[0].set_title(f'RGB + prompt points ({source})'); axes[0].axis('off')
    # Show spectral water candidate mask (NIR+thermal)
    mask_resized = cv2.resize(spectral_mask, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
    spec_overlay = rgb.copy()
    spec_overlay[mask_resized > 0] = (spec_overlay[mask_resized > 0] * 0.4 +
                                      np.array([255, 200, 0]) * 0.6).astype(np.uint8)
    axes[1].imshow(spec_overlay)
    axes[1].set_title('Spectral mask (yellow = NIR+thermal water candidate)'); axes[1].axis('off')
    axes[2].imshow(overlay); axes[2].set_title('TinySAM water mask'); axes[2].axis('off')
    plt.tight_layout()
    vis_path = str(Path(mask_path).with_suffix('')) + '_viz.jpg'
    plt.savefig(vis_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Visualisation -> {vis_path}')

    water_pct = 100 * best.mean()
    print(f'Water coverage: {water_pct:.1f}%')
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene_dir', help='Scene directory containing 5-band TIFF and RGB JPEG')
    parser.add_argument('--multi-point', type=int, default=5, metavar='N',
                        help='Number of NDWI-guided water prompt points (default: 5)')
    parser.add_argument('--output', default=None, help='Output mask path (default: <scene_dir>/tinysam_water_mask.png)')
    args = parser.parse_args()
    run(args.scene_dir, args.multi_point, args.output)


if __name__ == '__main__':
    main()
