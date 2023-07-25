_base_ = [
    '../_base_/models/efficientnet_b5.py',
    '../_base_/datasets/boiler_defects_640.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=456, backend='pillow', keep_ratio=True, interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=456, backend='pillow', keep_ratio=True, interpolation='bicubic'),
    dict(type='PackInputs'),
]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (64 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)

train_dataloader = dict(batch_size=4, dataset=dict(
    data_prefix='../boiler_defects_dataset_tiled_v1/train',
    pipeline=train_pipeline))
val_dataloader = dict(batch_size=4, dataset=dict(
    data_prefix='../boiler_defects_dataset_tiled_v1/valid',
    pipeline=test_pipeline))
test_dataloader = dict(batch_size=4, dataset=dict(
    data_prefix='../boiler_defects_dataset_tiled_v1/test',
    pipeline=test_pipeline))

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

# visualizer = dict(type='Visualizer', _delete_=True)
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(project='boiler_defects_classification',
                              name='efficientnet-b5_2xb4_in1k-456px_bd_tiled_v1',
                              ),
             define_metric_cfg=[dict(name='accuracy', step_metric='epoch'),
                                dict(name='recall', step_metric='epoch'),
                                dict(name='f1-score', step_metric='epoch'),
                                dict(name='precision', step_metric='epoch')]
             )
    ]
)
