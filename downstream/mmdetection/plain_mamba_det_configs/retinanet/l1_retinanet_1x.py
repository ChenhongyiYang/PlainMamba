_base_ = [
    '../template/retinanet_1x.py',
]

checkpoint_url = "https://huggingface.co/ChenhongyiYang/PlainMamba/resolve/main/l1.pth"

embed_dims = 192
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='PlainMambaAdapter',
        # Adapter Args
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        # PlainMamba args
        arch="L1",
        drop_path_rate=0.1,
        final_norm=False,
        convert_syncbn=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_url, prefix="backbone."),
    ),
    neck=dict(
        type='FPN',
        in_channels=[embed_dims, embed_dims, embed_dims, embed_dims],
        out_channels=256,
        add_extra_convs='on_output',
        start_level=1,
        num_outs=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
