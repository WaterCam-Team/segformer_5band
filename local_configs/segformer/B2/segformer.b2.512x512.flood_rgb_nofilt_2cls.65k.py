_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b2.pth',
    backbone=dict(
        type='mit_b2',
        in_chans=3,
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=0.5,
                 class_weight=[0.3, 0.7]),
            dict(type='LovaszLoss', loss_type='binary',
                 loss_weight=0.5, reduction='none'),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

dataset_type = 'dataset_rgb_nofilt'
data_root = 'data/modality_comparison/'

# REPLACE with values from: python tools/compute_band_stats.py --modality rgb_nofilt
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(640, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        img_ratios=[0.75, 1.0, 1.25],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=40,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='rgb_nofilt/img_dir/train',
            ann_dir='rgb_nofilt/ann_dir/train',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_nofilt/img_dir/val',
        ann_dir='rgb_nofilt/ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_nofilt/img_dir/test',
        ann_dir='rgb_nofilt/ann_dir/test',
        pipeline=test_pipeline))

runner = dict(type='IterBasedRunner', max_iters=65000)
evaluation = dict(interval=4000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=4000)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999),
                 weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={
                     'pos_block': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.),
                     'head': dict(lr_mult=10.),
                 }))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear', warmup_iters=1500, warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
