
_base_ = [
    'mmseg::_base_/models/segformer_mit-b0.py',
    'mmseg::_base_/default_runtime.py',
]

# Dataset settings
dataset_type = 'BaseSegDataset'
data_root    = '/content/processed/'
crop_size    = (640, 640)

# Training pipeline with sport-specific augmentations
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1920, 1080), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # Sport-specific augmentations via albumentations
    dict(
        type='Albu',
        transforms=[
            # Motion Blur: simulates camera panning during match
            dict(type='MotionBlur', blur_limit=9, p=0.4),
            # Gaussian Noise: simulates broadcast video compression
            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.3),
            # Color Jitter: simulates stadium lighting variations
            dict(type='RandomBrightnessContrast',
                 brightness_limit=0.3,
                 contrast_limit=0.3, p=0.4),
            # Color Jitter: simulates different cameras and artificial lights
            dict(type='HueSaturationValue',
                 hue_shift_limit=20,
                 sat_shift_limit=30,
                 val_shift_limit=20, p=0.3),
            # Occlusion Cutout: simulates players occluding the board
            dict(type='CoarseDropout',
                 max_holes=6,
                 max_height=80,
                 max_width=80,
                 min_holes=1,
                 fill_value=0, p=0.3),
        ],
        keymap={'img': 'image', 'gt_semantic_seg': 'mask'},
        update_pad_shape=False,
    ),
    dict(type='PackSegInputs'),
]

# Validation pipeline (no augmentation)
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

# Dataset meta info (2 classes: background and board)
meta = dict(
    classes=('background', 'board'),
    palette=[[0, 0, 0], [255, 0, 0]]
)

# Dataloaders
train_dataloader = dict(
    batch_size=4, num_workers=2,
    dataset=dict(
        type=dataset_type, data_root=data_root,
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        img_suffix='.jpeg', seg_map_suffix='.png',
        metainfo=meta, pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        type=dataset_type, data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        img_suffix='.jpeg', seg_map_suffix='.png',
        metainfo=meta, pipeline=val_pipeline,
    )
)

test_dataloader = val_dataloader

# Evaluation metrics
val_evaluator  = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

# Model: SegFormer-B1 architecture
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size,
        size_divisor=None,
    ),
    backbone=dict(
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/content/repo/checkpoints/mit_b1_imagenet.pth'
        )
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=2,
    ),
    test_cfg=dict(mode='whole')
)

# AdamW optimizer
optimizer = dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm':      dict(decay_mult=0.),
            'head':      dict(lr_mult=10.)
        }
    )
)

# LR scheduler with warmup
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0,    end=1500),
    dict(type='PolyLR',   eta_min=0.0,       power=1.0,      begin=1500, end=20000, by_epoch=False),
]

# Save best checkpoint based on mIoU
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000,
                    max_keep_ckpts=3, save_best='mIoU', rule='greater'),
    logger=dict(type='LoggerHook', interval=50),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')
