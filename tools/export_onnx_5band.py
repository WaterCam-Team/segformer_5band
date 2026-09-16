"""Export the 5-band SegFormer to ONNX and check it against torch.

Simpler than the type_pool model: a plain EncoderDecoder + SegFormerHead, no
prompt learner, no CLIP. The only thing pinning the graph to its traced
resolution is the head resizing _c4/_c3/_c2 to size=c1.size()[2:], which
mmseg.ops.resize turns into Python ints.
"""
import argparse, time
import mmcv, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor


def patch_resize_for_export():
    """scale_factor instead of size, for the trace only.

    The three resizes are exact power-of-two ratios (strides 32/16/8 against
    c1's 4), so this is equivalent *provided input dims divide by 32*. For
    other sizes ceil division makes them differ by a pixel, which is why the
    head itself is left alone.
    """
    from mmseg.models.decode_heads import segformer_head as head
    orig = head.resize

    def resize(input, size=None, scale_factor=None, mode='nearest',
               align_corners=None, warning=True):
        if size is not None and scale_factor is None:
            ih, iw = int(input.shape[2]), int(input.shape[3])
            oh, ow = int(size[0]), int(size[1])
            if ih and iw and oh % ih == 0 and ow % iw == 0 and oh // ih == ow // iw:
                return F.interpolate(input, scale_factor=float(oh // ih),
                                     mode=mode, align_corners=align_corners)
        return orig(input, size, scale_factor, mode, align_corners, warning)

    head.resize = resize
    return lambda: setattr(head, 'resize', orig)


class Wrap(nn.Module):
    def __init__(self, seg):
        super().__init__()
        self.seg = seg

    def forward(self, img):
        x = self.seg.extract_feat(img)
        out = self.seg._decode_head_forward_test(x, None)
        return F.interpolate(out, scale_factor=4, mode='bilinear',
                             align_corners=self.seg.align_corners)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bands', type=int, default=5)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--opset', type=int, default=12)
    args = ap.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cpu().eval()

    w0 = model.backbone.patch_embed1.proj.weight
    print(f'patch_embed1.proj.weight {tuple(w0.shape)}  -> {w0.shape[1]} input bands')
    assert w0.shape[1] == args.bands, 'band count mismatch'

    wrap = Wrap(model).eval()
    dummy = torch.randn(1, args.bands, args.height, args.width)

    unpatch = patch_resize_for_export()
    print(f'tracing at {args.height}x{args.width} ...', flush=True)
    with torch.no_grad():
        t0 = time.time()
        ref = wrap(dummy)
        print(f'  torch forward ok: {tuple(ref.shape)}  ({time.time()-t0:.1f}s)')
        torch.onnx.export(
            wrap, dummy, args.out, input_names=['input'],
            output_names=['logits'], opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes={'input': {2: 'h', 3: 'w'},
                          'logits': {2: 'h', 3: 'w'}})
    unpatch()
    print(f'exported -> {args.out}')

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
    got = sess.run(None, {'input': dummy.numpy()})[0]
    d = np.abs(got - ref.numpy())
    print(f'\nat traced size {args.height}x{args.width}:')
    print(f'  max abs logit diff : {d.max():.3e}')
    print(f'  argmax agreement   : '
          f'{(got.argmax(1) == ref.numpy().argmax(1)).mean()*100:.4f}%')

    h2, w2 = 384, 640
    print(f'\nat UNTRACED size {h2}x{w2}:')
    d2 = torch.randn(1, args.bands, h2, w2)
    with torch.no_grad():
        ref2 = wrap(d2).numpy()
    got2 = sess.run(None, {'input': d2.numpy()})[0]
    if got2.shape != ref2.shape:
        print(f'  SHAPE MISMATCH onnx={got2.shape} torch={ref2.shape}')
    else:
        print(f'  output shape       : {got2.shape}')
        print(f'  max abs logit diff : {np.abs(got2-ref2).max():.3e}')
        print(f'  argmax agreement   : '
              f'{(got2.argmax(1) == ref2.argmax(1)).mean()*100:.4f}%')


if __name__ == '__main__':
    main()
