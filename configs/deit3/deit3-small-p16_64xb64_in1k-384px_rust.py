_base_ = [
    '../_base_/models/deit3/deit3-small-p16-384.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(optimizer=dict(lr=1e-4, weight_decay=0.3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (64 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)


# ---- model settings ----
# Here we use init_cfg to load pre-trained model.
# In this way, only the weights of backbone will be loaded.
# And modify the num_classes to match our dataset.

model = dict(
    backbone=dict(
        img_size=640,
        init_cfg = dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_in21k-pre_3rdparty_in1k-384px_20221009-de116dd7.pth',
            prefix='backbone')
    ),
    head=dict(num_classes=4))

# ---- data settings ----
# We re-organized the dataset as `CustomDataset` format.
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=4,
    mean=[124.508, 116.050, 106.438],
    std=[58.577, 57.310, 57.437],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=640, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=320, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=320),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/rust_dataset/train',
        classes=['Heavy Visible Rust', 'No Rust', 'Slightly Visible Rust', 'Visible Rust'],
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/rust_dataset/valid',
        classes=['Heavy Visible Rust', 'No Rust', 'Slightly Visible Rust', 'Visible Rust'],
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/rust_dataset/test',
        classes=['Heavy Visible Rust', 'No Rust', 'Slightly Visible Rust', 'Visible Rust'],
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# Specify the evaluation metric for validation and testing.
val_evaluator = dict(type='Accuracy', topk=1)
test_evaluator = val_evaluator

# ---- schedule settings ----
# Usually in fine-tuning, we need a smaller learning rate and less training epochs.
# Specify the learning rate
# optim_wrapper=dict(
#    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
#    clip_grad=None,
# )
# Set the learning rate scheduler
# param_scheduler = dict(type='StepLR', by_epoch=True, step_size=1, gamma=0.1)

# Set training epochs and validate interval.
train_cfg = dict(by_epoch=True, max_epochs=22, val_interval=1)
# Use default settings for validation and testing
val_cfg = dict()
test_cfg = dict()

# ---- runtime settings ----
# Output training log every 10 iterations.
default_hooks = dict(logger=dict(interval=10),
                     checkpoint = dict(type='CheckpointHook', interval=10))

# If you want to ensure reproducibility, set a random seed. And enable the
# deterministic option in cuDNN to further ensure reproducibility, but it may
# reduce the training speed.
randomness = dict(seed=0, deterministic=False)

visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)
