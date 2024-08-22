_base_ = './segmenter_vit-b_mask_8xb1-80k_pascal_voc12_aug-512x512.py'

model = dict(
    backbone=dict(
        out_indices=[11]
    ),
    decode_head=dict(
        _delete_=True,
        type='FCNHead',
        in_index=[0],
        in_channels=[768],
        channels=768,
        input_transform='resize_concat',
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=21,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))