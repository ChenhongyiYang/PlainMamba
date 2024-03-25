# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales

from mmcls.plain_mamba_dev.models.necks.custom_avg_pooling import CustomAveragePooling

__all__ = ['GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales','CustomAveragePooling']
