import argparse
import math
import random
import shutil
import sys
import yaml
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import VideoFolder
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import video_models
from pytorch_msssim import ms_ssim
from ipywidgets import interact, widgets


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, config):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": config['learning_rate']},
        "aux": {"type": "Adam", "lr": config['aux_learning_rate']},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


# TODO:image2video
def compute_psnr(a, b):
    T = len(a)
    tensor_a = a[0]  
    tensor_b = b[0]
    for i in range(T-1):
        tensor_a = torch.cat([tensor_a, a[i+1]], dim=0)
        tensor_b = torch.cat([tensor_b, b[i+1]], dim=0)
    mse = torch.mean((tensor_a - tensor_b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    # 将时间维度合并到批次维度以处理视频帧
    T = len(a)
    tensor_a = a[0]  
    tensor_b = b[0]
    for i in range(T-1):
        tensor_a = torch.cat([tensor_a, a[i+1]], dim=0)
        tensor_b = torch.cat([tensor_b, b[i+1]], dim=0)
    return ms_ssim(tensor_a, tensor_b, data_range=1.).item()

def compute_bpp(out_net):
    # 假设 x_hat 的形状为 [时间, 批次, 通道, 高度, 宽度]
    T = len(out_net['x_hat'])
    B, C, H, W = out_net['x_hat'][0].shape
    num_pixels = T * B * H * W  # 批次 * 时间 * 高度 * 宽度
    tensor_likelihoods = out_net['likelihoods'][0]
    for i in range(T-1):
        tensor_likelihoods = torch.cat([tensor_likelihoods, out_net['likelihoods'][i+1]], dim=0)
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in out_net['likelihoods'].values()).item()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml config"
    )
    args = parser.parse_args(argv)
    return args


def readyaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
        config = yaml.load(data, Loader=yaml.FullLoader)
    return config


def test_epoch(test_dataloader, model, metrics):
    model.eval()
    device = next(model.parameters()).device

    org_seq = RawVideoSequence.from_file(str(sequence))
    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    with torch.no_grad():
        for batch in test_dataloader:
            d = [frames.to(device) for frames in batch]
            out_net = model.forward(d) # outnet:dict_keys(['x_hat', 'likelihoods'])
            # out_net['x_hat'].clamp_(0, 1) # TODO:confirm
            print(out_net.keys())

            # compute metrics TODO:use eval_model as api
            metrics['psnr'].update(compute_psnr(d, out_net["x_hat"]))
            metrics['ms_ssim'].update(compute_msssim(d, out_net["x_hat"]))
            metrics['rate'].update(compute_bpp(out_net))

        print(f"PSNR: {metrics['psnr'].avg:.2f}dB")
        print(f"MS-SSIM: {metrics['ms_ssim'].avg:.4f}")
        print(f"Bit-rate: {metrics['rate'].avg:.3f} bpp")

    return metrics


def main(argv):
    args = parse_args(argv)
    config = readyaml(args.config)
    device = "cuda" if config['cuda'] and torch.cuda.is_available() else "cpu"

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])

    # Warning, the order of the transform composition should be kept.
    patch_size = (config['patch_size'], config['patch_size'])

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.CenterCrop(patch_size)]
    )
    test_dataset = VideoFolder(
        config['dataset'],
        rnd_interval=False,
        rnd_temp_order=False,
        split="test",
        transform=test_transforms,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['test_batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = video_models[config['model']](quality=3, pretrained=True).eval().to(device)

    if config['checkpoint']:  # load from previous checkpoint #TODO:assert,must have a checkpoint?
        print("Loading", config['checkpoint'])
        checkpoint = torch.load(config['checkpoint'], map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

    psnr = AverageMeter()
    ms_ssim = AverageMeter()
    rate = AverageMeter()
    metrics = {'psnr':psnr, 'ms_ssim':ms_ssim, 'rate':rate}

    metrics = test_epoch(test_dataloader, net, metrics)

if __name__ == "__main__":
    main(sys.argv[1:])