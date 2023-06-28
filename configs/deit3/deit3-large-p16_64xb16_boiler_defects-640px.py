_base_ = [
    '../_base_/models/deit3/deit3-large-p16-384.py',
    '../_base_/datasets/boiler_defects_640.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(optimizer=dict(lr=1e-5, weight_decay=0.2))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (64 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=2)

# ---- model settings ----
# Here we use init_cfg to load pre-trained model.
# In this way, only the weights of backbone will be loaded.
# And modify the num_classes to match our dataset.

model = dict(
    backbone=dict(
        img_size=640,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/deit3/deit3-large-p16_in21k-pre_3rdparty_in1k-384px_20221009-75fea03f.pth',
            prefix='backbone')
    ),
    head=dict(num_classes=2))

# Specify the evaluation metric for validation and testing.
# val_evaluator = dict(type='Accuracy', topk=1)
# val_evaluator = dict(type='Accuracy', topk=1)
# val_evaluator = dict(_delete_=True, type='SingleLabelMetric', average=None)
val_evaluator = [
    dict(type='Accuracy', topk=1),
    dict(type='MultiLabelMetric',
         items=['precision', 'recall', 'f1-score'],
         average=None,
         thr=0.5),
]
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
train_cfg = dict(by_epoch=True, max_epochs=15, val_interval=1)
# Use default settings for validation and testing
val_cfg = dict()
test_cfg = dict()

# ---- runtime settings ----
# Output training log every 10 iterations.

default_hooks = dict(
    logger=dict(interval=1),

    checkpoint=dict(type='CheckpointHook', interval=5,
                    max_keep_ckpts=5, save_best="auto"
                    ),

    visualization=dict(type='VisualizationHook', enable=True)
)

# If you want to ensure reproducibility, set a random seed. And enable the
# deterministic option in cuDNN to further ensure reproducibility, but it may
# reduce the training speed.
randomness = dict(seed=0, deterministic=False)

visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(project='boiler_defects_classification',
                              name='deit3-large-p16_64xb16_boiler_defects-640px',
                              ),
             define_metric_cfg=[dict(name='accuracy', step_metric='epoch'),
                                dict(name='recall', step_metric='epoch'),
                                dict(name='f1-score', step_metric='epoch'),
                                dict(name='precision', step_metric='epoch')]
             )
    ]
)
