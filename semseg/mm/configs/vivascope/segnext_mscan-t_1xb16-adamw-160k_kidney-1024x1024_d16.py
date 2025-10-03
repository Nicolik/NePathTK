checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'
crop_size = (
    1024,
    1024,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        1024,
        1024,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = 'E:/AnnotationsVivaScope/v2'
dataset_type = 'KidneyDataset_d16'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=16000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
ham_norm_cfg = dict(num_groups=32, requires_grad=True, type='GN')
img_ratios = [
    0.75,
    1.0,
    1.25,
]
launcher = 'none'
load_from = 'E:/Users/Altini/PycharmProjects/mmsegmentation/work_dirs\\segnext_mscan-t_1xb16-adamw-160k_kidney-1024x1024_CMC\\iter_160000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attention_kernel_paddings=[
            2,
            [
                0,
                3,
            ],
            [
                0,
                5,
            ],
            [
                0,
                10,
            ],
        ],
        attention_kernel_sizes=[
            5,
            [
                1,
                7,
            ],
            [
                1,
                11,
            ],
            [
                1,
                21,
            ],
        ],
        depths=[
            3,
            3,
            5,
            2,
        ],
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=[
            32,
            64,
            160,
            256,
        ],
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth',
            type='Pretrained'),
        mlp_ratios=[
            8,
            8,
            4,
            4,
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        type='MSCAN'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            1024,
            1024,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        ham_channels=256,
        ham_kwargs=dict(
            MD_R=16,
            MD_S=1,
            eval_steps=7,
            inv_t=100,
            rand_init=True,
            train_steps=6),
        in_channels=[
            64,
            160,
            256,
        ],
        in_index=[
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=4,
        type='LightHamHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=160000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(
            img_path='Validation/CMC/Image',
            seg_map_path='Validation/CMC/Mask'),
        data_root='E:/AnnotationsVivaScope/v2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KidneyDataset_d16'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=16000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='Train/CMC/Image', seg_map_path='Train/CMC/Mask'),
        data_root='E:/AnnotationsVivaScope/v2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KidneyDataset_d16'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(
    module=dict(
        backbone=dict(
            act_cfg=dict(type='GELU'),
            attention_kernel_paddings=[
                2,
                [
                    0,
                    3,
                ],
                [
                    0,
                    5,
                ],
                [
                    0,
                    10,
                ],
            ],
            attention_kernel_sizes=[
                5,
                [
                    1,
                    7,
                ],
                [
                    1,
                    11,
                ],
                [
                    1,
                    21,
                ],
            ],
            depths=[
                3,
                3,
                5,
                2,
            ],
            drop_path_rate=0.1,
            drop_rate=0.0,
            embed_dims=[
                32,
                64,
                160,
                256,
            ],
            init_cfg=dict(
                checkpoint=
                'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth',
                type='Pretrained'),
            mlp_ratios=[
                8,
                8,
                4,
                4,
            ],
            norm_cfg=dict(requires_grad=True, type='BN'),
            type='MSCAN'),
        data_preprocessor=dict(
            bgr_to_rgb=True,
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            pad_val=0,
            seg_pad_val=255,
            size=(
                1024,
                1024,
            ),
            std=[
                58.395,
                57.12,
                57.375,
            ],
            test_cfg=dict(size_divisor=32),
            type='SegDataPreProcessor'),
        decode_head=dict(
            align_corners=False,
            channels=256,
            dropout_ratio=0.1,
            ham_channels=256,
            ham_kwargs=dict(
                MD_R=16,
                MD_S=1,
                eval_steps=7,
                inv_t=100,
                rand_init=True,
                train_steps=6),
            in_channels=[
                64,
                160,
                256,
            ],
            in_index=[
                1,
                2,
                3,
            ],
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
            num_classes=4,
            type='LightHamHead'),
        pretrained=None,
        test_cfg=dict(mode='whole'),
        train_cfg=dict(),
        type='EncoderDecoder'),
    type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.9, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.1, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='Validation/CMC/Image',
            seg_map_path='Validation/CMC/Mask'),
        data_root='E:/AnnotationsVivaScope/v2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KidneyDataset_d16'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir=
    'D:\\Datasets\\KIDNEY\\SegmentationOutputVivaScope\\pairwise_visualizer\\segnext_mscan-t_1xb16-adamw\\CMC',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs\\segnext_mscan-t_1xb16-adamw-160k_kidney-1024x1024_CMC'
