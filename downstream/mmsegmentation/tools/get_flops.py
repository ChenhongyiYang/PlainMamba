import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

from mmseg.models import build_segmentor
import torch

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info
from mmseg.models.builder import build_segmentor
import numpy as np

from mmcls.models import build_classifier

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--is-mamba', action='store_true')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 2048],
        help='input image size')
    args = parser.parse_args()
    return args

def sra_flops(h, w, r, dim, num_heads):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = h / r * w / r

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads

    return f1 + f2


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



def get_no_attn_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    return flops, params

def get_tr_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    net = net.backbone
    try:
        stage1 = sra_flops(H // 4, W // 4,
                           net.block1[0].attn.sr_ratio,
                           net.block1[0].attn.dim,
                           net.block1[0].attn.num_heads) * len(net.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           net.block2[0].attn.sr_ratio,
                           net.block2[0].attn.dim,
                           net.block2[0].attn.num_heads) * len(net.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           net.block3[0].attn.sr_ratio,
                           net.block3[0].attn.dim,
                           net.block3[0].attn.num_heads) * len(net.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           net.block4[0].attn.sr_ratio,
                           net.block4[0].attn.dim,
                           net.block4[0].attn.num_heads) * len(net.block4)
    except:
        stage1 = sra_flops(H // 4, W // 4,
                           net.block1[0].attn.squeeze_ratio,
                           64,
                           net.block1[0].attn.num_heads) * len(net.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           net.block2[0].attn.squeeze_ratio,
                           128,
                           net.block2[0].attn.num_heads) * len(net.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           net.block3[0].attn.squeeze_ratio,
                           320,
                           net.block3[0].attn.num_heads) * len(net.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           net.block4[0].attn.squeeze_ratio,
                           512,
                           net.block4[0].attn.num_heads) * len(net.block4)

    print(stage1 + stage2 + stage3 + stage4)
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    if args.is_mamba:
        flops, params = get_no_attn_flops(model, input_shape)
        flops = flops / 1e9
        params = params / 1e6
    else:
        flops, params = get_tr_flops(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\n')
    if args.is_mamba:
        ssm_flops = get_ssm_flops(tuple(args.shape), model)
        print("SSM:", ssm_flops, "GFLOPs")
        print("SSM FLOPs: %.2fG" % ssm_flops)
        print("Total FLOPs: %.2fG" % (flops + ssm_flops))
    print(f'Input shape: {input_shape}\n'
          f'Flops: {flops}G\nParams: {params}\n{split_line}M')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()