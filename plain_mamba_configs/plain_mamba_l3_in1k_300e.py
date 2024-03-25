_base_ = [
    '../configs/_base_/datasets/imagenet_bs64_swin_224_lmdb.py',
    '../configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../configs/_base_/default_runtime.py'
]


# model settings
embed_dims = 448

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PlainMamba',
        arch="L3",
        drop_path_rate=0.4,
    ),
    neck=dict(
        type='CustomAveragePooling'
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=embed_dims,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'
        ),
        topk=(1, 5)
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(
        augments=[
            dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
            dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
        ]
    )
)

# ------------------------------------------------------------
# data settings
samples_per_gpu=256
data = dict(samples_per_gpu=samples_per_gpu, workers_per_gpu=4)

base_bs = 512
base_lr = 5e-4

# opt settings
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0),
        '.A_log': dict(decay_mult=0.0),
        '.D': dict(decay_mult=0.0),
        '.direction_Bs': dict(decay_mult=0.0),
    }
)
world_size = 4
optimizer = dict(
    lr=5e-4 * samples_per_gpu * world_size / 512,
    paramwise_cfg=paramwise_cfg)
lr_config = dict(warmup_iters=20)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# other running settings
save_interval = 5
checkpoint_config = dict(interval=save_interval, max_keep_ckpts=10)
evaluation = dict(interval=save_interval, metric='accuracy')
custom_hooks = [
    dict(
        type='EMACheckpointHook',
        momentum=1e-4,
        priority='ABOVE_NORMAL',
        save_interval=save_interval,
        max_keep_ckpts=10,
        decay_epochs=(250,),
        decay_factor=10.,
        resume_from=None,
    )
]
fp16 = None  # make sure fp16 (mm version) is None when using AMP optimizer
runner = dict(type='EpochBasedRunner')
