import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class dataset_rgb_std(CustomDataset):
    """Standard RGB (NIR-filter-on) water segmentation dataset.

    Images are JPG files from the NIR-OFF camera channel (standard visible
    light). Labels are single-channel PNGs: 0=background, 1=water.
    """

    CLASSES = ('background', 'water')
    PALETTE = [[120, 120, 120], [204, 0, 102]]

    def __init__(self, **kwargs):
        super(dataset_rgb_std, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)


@DATASETS.register_module()
class dataset_rgb_nofilt(CustomDataset):
    """No-NIR-filter RGB water segmentation dataset.

    Images are JPG files from the NIR-ON camera channel (NIR bleeds into all
    colour channels, particularly red). Labels are single-channel PNGs:
    0=background, 1=water.
    """

    CLASSES = ('background', 'water')
    PALETTE = [[120, 120, 120], [204, 0, 102]]

    def __init__(self, **kwargs):
        super(dataset_rgb_nofilt, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
