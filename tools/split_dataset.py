#!/usr/bin/env python
"""Organise co-registered scene directories into the modality comparison dataset.

Expects each scene directory to contain:
    *-NIR-OFF.jpg               -> standard RGB
    *-NIR-ON.jpg                -> no-filter RGB
    *.pgm                       -> LWIR thermal
    color_preserved_5_band.tiff -> 5-band fused (R,G,B,Thermal,NIR)
    water_mask.png              -> ground-truth label (0=background, 1=water)

Produces:
    <output>/
      rgb_std/img_dir/{train,val,test}/   <- NIR-OFF JPGs
      rgb_std/ann_dir/{train,val,test}/   <- water_mask PNGs
      rgb_nofilt/img_dir/{...}/           <- NIR-ON JPGs
      rgb_nofilt/ann_dir/{...}/
      lwir/img_dir/{...}/                 <- PGM files
      lwir/ann_dir/{...}/
      fiveband/img_dir/{...}/             <- 5-band TIFFs
      fiveband/ann_dir/{...}/

Files are copied (not symlinked) to keep datasets self-contained.

Usage:
    python tools/split_dataset.py /path/to/coreg/root /path/to/output/dataset
        [--train 0.70] [--val 0.15] [--test 0.15] [--seed 42]
"""
import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

MODALITIES = {
    'rgb_std':   {'suffix': '.jpg',  'pattern': '*-NIR-OFF.jpg'},
    'rgb_nofilt':{'suffix': '.jpg',  'pattern': '*-NIR-ON.jpg'},
    'lwir':      {'suffix': '.pgm',  'pattern': '*.pgm'},
    'fiveband':  {'suffix': '.tif',  'pattern': 'color_preserved_5_band.tiff'},
}


def find_scenes(root: Path) -> list[Path]:
    """Return scene dirs that have all modality files AND a water_mask.png."""
    scenes = []
    for mask in sorted(root.rglob('water_mask.png')):
        d = mask.parent
        has_all = (
            any(d.glob('*-NIR-OFF.jpg')) and
            any(d.glob('*-NIR-ON.jpg')) and
            any(d.glob('*.pgm')) and
            (d / 'color_preserved_5_band.tiff').exists()
        )
        if has_all:
            scenes.append(d)
        else:
            log.warning(f'Skipping incomplete scene: {d}')
    return scenes


def copy_scene(scene: Path, split: str, output: Path, idx: int):
    """Copy all modality files for one scene into the output dataset."""
    stem = f'scene_{idx:05d}'

    # RGB std
    src = next(scene.glob('*-NIR-OFF.jpg'))
    dst_dir = output / 'rgb_std' / 'img_dir' / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / f'{stem}.jpg')

    # RGB no-filter
    src = next(scene.glob('*-NIR-ON.jpg'))
    dst_dir = output / 'rgb_nofilt' / 'img_dir' / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / f'{stem}.jpg')

    # LWIR
    pgms = list(scene.glob('*.pgm'))
    if pgms:
        dst_dir = output / 'lwir' / 'img_dir' / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pgms[0], dst_dir / f'{stem}.pgm')

    # 5-band TIFF
    dst_dir = output / 'fiveband' / 'img_dir' / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene / 'color_preserved_5_band.tiff', dst_dir / f'{stem}.tif')

    # Label (shared across all modalities)
    mask_src = scene / 'water_mask.png'
    for modality in MODALITIES:
        dst_dir = output / modality / 'ann_dir' / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mask_src, dst_dir / f'{stem}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', help='Root of co-registered scene directories')
    parser.add_argument('output', help='Output dataset directory')
    parser.add_argument('--train', type=float, default=0.70)
    parser.add_argument('--val',   type=float, default=0.15)
    parser.add_argument('--test',  type=float, default=0.15)
    parser.add_argument('--seed',  type=int,   default=42)
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        'train + val + test must sum to 1.0'

    scenes = find_scenes(Path(args.root))
    if not scenes:
        log.error('No complete scenes found (need water_mask.png + all modality files)')
        return
    log.info(f'Found {len(scenes)} labelled scenes')

    random.seed(args.seed)
    random.shuffle(scenes)
    n = len(scenes)
    n_train = int(n * args.train)
    n_val   = int(n * args.val)
    splits = (
        [('train', s) for s in scenes[:n_train]] +
        [('val',   s) for s in scenes[n_train:n_train + n_val]] +
        [('test',  s) for s in scenes[n_train + n_val:]]
    )

    output = Path(args.output)
    for idx, (split, scene) in enumerate(splits):
        copy_scene(scene, split, output, idx)

    counts = {'train': n_train, 'val': n_val, 'test': n - n_train - n_val}
    log.info(f'Dataset written to {output}')
    for split, count in counts.items():
        log.info(f'  {split}: {count} scenes')
    log.info('Next: run tools/compute_band_stats.py to get normalization values')


if __name__ == '__main__':
    main()
