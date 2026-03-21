
_base_ = [
    'mmseg::_base_/models/segformer_mit-b1.py',
    'mmseg::_base_/default_runtime.py',
]

# Dataset settings
dataset_type = 'BaseSegDataset'
data_root    = '/content/processed/'
crop_size    = (640, 640)

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1920, 1080), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

# Validation pipeline
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

# Model: 2 classes with pretrained backbone
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size,        # explicitly set size
        size_divisor=None,     # explicitly unset size_divisor
    ),
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/content/repo/checkpoints/mit_b0_imagenet.pth'
        )
    ),
    decode_head=dict(num_classes=2),
    test_cfg=dict(mode='whole')
)

# AdamW optimizer - explicitly override base config to prevent momentum inheritance
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
    dict(type='PolyLR',   eta_min=0.0,       power=1.0,      begin=1500, end=160000, by_epoch=False),
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
