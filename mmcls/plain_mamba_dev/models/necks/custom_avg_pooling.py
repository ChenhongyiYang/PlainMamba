import torch
import torch.nn as nn

from mmcls.models.builder import NECKS


@NECKS.register_module()
class CustomAveragePooling(nn.Module):
    def __init__(self, dim=2):
        super(CustomAveragePooling, self).__init__()

    def init_weights(self):
        pass

    def forward(self, inputs):
        assert isinstance(inputs, tuple)
        assert isinstance(inputs[0], torch.Tensor)
        final_features = inputs[0]

        assert len(final_features.shape) == 3
        final_features = torch.mean(final_features, dim=1)
        return final_features

