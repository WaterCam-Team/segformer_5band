#!/usr/bin/env python
"""Run inference with all four trained modality models and compare results.

Evaluates each model on the shared test split and produces a comparison table
of mIoU, water IoU, F1, precision, and recall.

Usage:
    python tools/compare_modalities.py \\
        --data_root data/modality_comparison \\
        --checkpoints \\
            work_dirs/rgb_std/latest.pth \\
            work_dirs/rgb_nofilt/latest.pth \\
            work_dirs/lwir/latest.pth \\
            work_dirs/fiveband/latest.pth \\
        --configs \\
            local_configs/segformer/B2/segformer.b2.512x512.flood_rgb_std_2cls.65k.py \\
            local_configs/segformer/B2/segformer.b2.512x512.flood_rgb_nofilt_2cls.65k.py \\
            local_configs/segformer/B2/segformer.b2.512x512.flood_lwir_2cls.65k.py \\
            local_configs/segformer/B2/segformer.b2.512x512.flood_5band_2cls.65k.py \\
        [--device cpu]
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

MODALITY_NAMES = ['rgb_std', 'rgb_nofilt', 'lwir', 'fiveband']


def compute_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 2) -> dict:
    """Compute per-class and mean IoU, F1, precision, recall."""
    metrics = {}
    ious, f1s, precs, recs = [], [], [], []
    for cls in range(num_classes):
        tp = ((pred == cls) & (gt == cls)).sum()
        fp = ((pred == cls) & (gt != cls)).sum()
        fn = ((pred != cls) & (gt == cls)).sum()
        iou  = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        ious.append(float(iou))
        f1s.append(float(f1))
        precs.append(float(prec))
        recs.append(float(rec))
        metrics[f'iou_cls{cls}'] = float(iou)

    metrics['mIoU']          = float(np.mean(ious))
    metrics['water_iou']     = ious[1]
    metrics['water_f1']      = f1s[1]
    metrics['water_prec']    = precs[1]
    metrics['water_recall']  = recs[1]
    return metrics


def evaluate_model(config_path: str, checkpoint_path: str,
                   data_root: str, device: str) -> dict:
    """Load model, run inference on test split, return aggregate metrics."""
    import mmcv
    from mmseg.apis import init_segmentor, inference_segmentor
    from mmseg.datasets import build_dataset
    from mmcv.image import tensor2imgs

    cfg = mmcv.Config.fromfile(config_path)
    cfg.data.test.test_mode = True

    model = init_segmentor(config_path, checkpoint_path, device=device)
    model.eval()

    dataset = build_dataset(cfg.data.test)
    log.info(f'  test set: {len(dataset)} images')

    all_preds, all_gts = [], []
    for i, data in enumerate(dataset):
        img_path = data['img_metas'][0].data['filename']
        result = inference_segmentor(model, img_path)
        pred = result[0]

        ann_path = dataset.get_ann_info(i)['seg_map']
        import cv2
        gt = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        if gt.shape != pred.shape:
            import cv2 as _cv2
            gt = _cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                             interpolation=_cv2.INTER_NEAREST)
        all_preds.append(pred.ravel())
        all_gts.append(gt.ravel())

    preds = np.concatenate(all_preds)
    gts   = np.concatenate(all_gts)
    return compute_metrics(preds, gts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root',   required=True)
    parser.add_argument('--configs',     nargs=4, required=True, metavar='CONFIG')
    parser.add_argument('--checkpoints', nargs=4, required=True, metavar='CKPT')
    parser.add_argument('--device',      default='cpu')
    args = parser.parse_args()

    results = {}
    for name, cfg, ckpt in zip(MODALITY_NAMES, args.configs, args.checkpoints):
        log.info(f'\nEvaluating {name} ...')
        metrics = evaluate_model(cfg, ckpt, args.data_root, args.device)
        results[name] = metrics
        log.info(f'  mIoU={metrics["mIoU"]:.4f}  water_IoU={metrics["water_iou"]:.4f}  '
                 f'F1={metrics["water_f1"]:.4f}  P={metrics["water_prec"]:.4f}  '
                 f'R={metrics["water_recall"]:.4f}')

    # Print comparison table
    header = f'{"Modality":<14} {"mIoU":>7} {"Water IoU":>10} {"F1":>7} {"Precision":>10} {"Recall":>8}'
    print('\n' + '=' * len(header))
    print(header)
    print('-' * len(header))
    for name, m in results.items():
        print(f'{name:<14} {m["mIoU"]:>7.4f} {m["water_iou"]:>10.4f} '
              f'{m["water_f1"]:>7.4f} {m["water_prec"]:>10.4f} {m["water_recall"]:>8.4f}')
    print('=' * len(header))

    # Save to CSV
    import csv
    out_csv = Path('work_dirs/modality_comparison.csv')
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['modality'] + list(next(iter(results.values())).keys()))
        w.writeheader()
        for name, m in results.items():
            w.writerow({'modality': name, **m})
    log.info(f'\nResults saved to {out_csv}')


if __name__ == '__main__':
    main()
