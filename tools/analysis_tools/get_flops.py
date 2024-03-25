# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info

from mmcls.models import build_classifier
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--is-mamba', action='store_true')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args

def get_ssm_flops(input_shape, model):
    # Reference: https://github.com/state-spaces/mamba/issues/110
    directions = 4
    depth = model.backbone.num_layers
    patch_size = model.backbone.patch_size
    embed_dims = model.backbone.embed_dims

    L = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    d_inner = embed_dims * model.backbone.layer_cfgs["mamba_cfg"]["expand"]
    d_state = model.backbone.layer_cfgs["mamba_cfg"]["d_state"]
    flops = d_inner * d_state * 9 * L * directions * depth
    flops += d_inner * L * directions * depth  # D

    gflops = flops / 1e9
    return gflops


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_classifier(cfg.model).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)
    flops = flops / 1e9
    params = params / 1e6

    split_line = '=' * 30
    print(f'{split_line}')
    if args.is_mamba:
        ssm_flops = get_ssm_flops(tuple(args.shape), model)
        print("SSM FLOPs: %.2fG" % ssm_flops)
        print("Total FLOPs: %.2fG" % (flops + ssm_flops))

    print(f'Input shape: {input_shape}\n'
          f'Flops: {flops}G\nParams: {params}M\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

if __name__ == '__main__':
    main()
