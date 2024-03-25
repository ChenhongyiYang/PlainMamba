_base_ = [
    '../../configs/_base_/datasets/coco_detection.py',
    '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py'
]

# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline))

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
    paramwise_cfg=dict(
    custom_keys={
        'level_embed': dict(decay_mult=0.),
        'pos_embed': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'bias': dict(decay_mult=0.),
        '.pos_embed': dict(decay_mult=0.0),
        '.A_log': dict(decay_mult=0.0),
        '.D': dict(decay_mult=0.0),
        '.direction_Bs': dict(decay_mult=0.0),
    }))
# find_unused_parameters = True
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
fp16 = None #  dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)
find_unused_parameters = True
