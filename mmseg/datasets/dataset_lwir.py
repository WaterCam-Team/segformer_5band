import os.path as osp

from .builder import DATASETS
from .custom_5band import CustomDataset_5band


@DATASETS.register_module()
class dataset_lwir(CustomDataset_5band):
    """LWIR (thermal) water segmentation dataset.

    Images are PGM files from the FLIR Lepton sensor, loaded as single-channel
    grayscale. Labels are single-channel PNGs: 0=background, 1=water.
    """

    CLASSES = ('background', 'water')
    PALETTE = [[120, 120, 120], [204, 0, 102]]

    def __init__(self, **kwargs):
        super(dataset_lwir, self).__init__(
            img_suffix='.pgm',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
