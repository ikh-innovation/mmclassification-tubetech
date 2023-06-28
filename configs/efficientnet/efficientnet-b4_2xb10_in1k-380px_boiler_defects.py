_base_ = [
    '../_base_/models/efficientnet_b4.py',
    '../_base_/datasets/boiler_defects_640.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=380),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=380),
    dict(type='PackClsInputs'),
]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (64 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=20)

train_dataloader = dict(batch_size=10, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=10, dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(batch_size=10, dataset=dict(pipeline=test_pipeline))

# model settings
model = dict(
    head=dict(
        num_classes=2,
        topk=1
    ))

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5,
                                     max_keep_ckpts=5, save_best="auto"))
# logger = dict(interval=1),
# visualization = dict(type='VisualizationHook', enable=True)


visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(project='boiler_defects_classification',
                              name='efficientnet-b4_2xb10_in1k-380px',
                              ),
             define_metric_cfg=[dict(name='accuracy', step_metric='epoch'),
                                dict(name='recall', step_metric='epoch'),
                                dict(name='f1-score', step_metric='epoch'),
                                dict(name='precision', step_metric='epoch')]
             )
    ]
)