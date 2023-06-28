
# ---- data settings ----
# We re-organized the dataset as `CustomDataset` format.
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=2)
# mean=[124.508, 116.050, 106.438],
# std=[58.577, 57.310, 57.437],
# to_rgb=True,


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=640, backend='pillow', keep_ratio=True, interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=640, backend='pillow', keep_ratio=True, interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=320),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='../boiler_defects_dataset2/train',
        classes=['clean', 'defect'],
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='../boiler_defects_dataset2/valid',
        classes=['clean', 'defect'],
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='../boiler_defects_dataset2/test',
        classes=['clean', 'defect'],
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# Specify the evaluation metric for validation and testing.
val_evaluator = dict(type='Accuracy', topk=1)
test_evaluator = val_evaluator
