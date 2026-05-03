#!/usr/bin/env python
"""Compute per-band mean and std over the training split of a modality.

Run this after split_dataset.py to get the normalization values to substitute
into the img_norm_cfg of each training config.

Usage:
    python tools/compute_band_stats.py \\
        --data_root data/modality_comparison \\
        --modality rgb_std          # or rgb_nofilt | lwir | fiveband

Output (paste into config img_norm_cfg):
    mean=[R, G, B]    std=[R, G, B]          # for 3-band modalities
    mean=[...5...]    std=[...5...]           # for fiveband
    mean=X            std=X                  # for lwir
"""
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

SUFFIX = {
    'rgb_std':    '.jpg',
    'rgb_nofilt': '.jpg',
    'lwir':       '.pgm',
    'fiveband':   '.tif',
}


def load_image(path: Path, modality: str) -> np.ndarray:
    """Return image as float32 array of shape (H, W, C)."""
    if modality == 'fiveband':
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32).transpose(1, 2, 0)
        return img
    elif modality == 'lwir':
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img[:, :, np.newaxis].astype(np.float32)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root', required=True,
                        help='Root of modality comparison dataset')
    parser.add_argument('--modality', required=True,
                        choices=['rgb_std', 'rgb_nofilt', 'lwir', 'fiveband'])
    args = parser.parse_args()

    img_dir = Path(args.data_root) / args.modality / 'img_dir' / 'train'
    suffix = SUFFIX[args.modality]
    files = sorted(img_dir.rglob(f'*{suffix}'))
    if not files:
        log.error(f'No {suffix} files found in {img_dir}')
        return
    log.info(f'Computing stats over {len(files)} training images ({args.modality})')

    # Online mean/variance (Welford not needed at this scale; accumulate sums)
    first = load_image(files[0], args.modality)
    n_bands = first.shape[2]
    pixel_sum   = np.zeros(n_bands, dtype=np.float64)
    pixel_sum_sq = np.zeros(n_bands, dtype=np.float64)
    pixel_count = 0

    for path in files:
        img = load_image(path, args.modality)
        for b in range(n_bands):
            band = img[:, :, b].ravel()
            pixel_sum[b]    += band.sum()
            pixel_sum_sq[b] += (band ** 2).sum()
            if b == 0:
                pixel_count += len(band)

    means = pixel_sum / pixel_count
    stds  = np.sqrt(pixel_sum_sq / pixel_count - means ** 2)

    log.info(f'Band stats for {args.modality} (training set):')
    for b in range(n_bands):
        log.info(f'  band {b+1}: mean={means[b]:.4f}  std={stds[b]:.4f}')

    # Print paste-ready config snippet
    print('\n--- paste into config img_norm_cfg ---')
    if args.modality == 'lwir':
        print(f'img_norm_cfg = dict(mean={means[0]:.4f}, std={stds[0]:.4f})')
    else:
        m = ', '.join(f'{v:.4f}' for v in means)
        s = ', '.join(f'{v:.4f}' for v in stds)
        if args.modality == 'fiveband':
            print(f'img_norm_cfg = dict(mean=[{m}], std=[{s}], to_rgb=False)')
        else:
            print(f'img_norm_cfg = dict(mean=[{m}], std=[{s}], to_rgb=True)')
    print('--------------------------------------\n')


if __name__ == '__main__':
    main()
