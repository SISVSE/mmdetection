# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
from os import path as osp

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    model.show_result(
        args.img,
        result,
        out_file='_'.join([args.img.split('.jpg')[0], 'result.jpg']),
        score_thr=args.score_thr)
    print('_'.join([args.img.split('.jpg')[0], 'result.jpg']))


if __name__ == '__main__':
    args = parse_args()
    main(args)
