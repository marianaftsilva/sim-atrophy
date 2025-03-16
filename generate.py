#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:41:34 2022

@author: mds19
"""

import torch
import torch.utils.data
from argparse import ArgumentParser

from utils.dataloader import DataSetEval, get_transforms
from training.epoch_cycles import inference_cycle
import os


def get_args_from_command_line():
    parser = ArgumentParser(description="Parser for testing Sim Atrophy")
    parser.add_argument("--region-list-path", dest="region_list_path", default="./region_list.npy")
    parser.add_argument("--atr-path", dest="atr_folder", default="/data/ADNI_mariana/Train/")
    parser.add_argument("--seg-path", dest="seg_folder", default="/data/ADNI_mariana/Seg_down/")
    parser.add_argument("--parc-path", dest="parc_folder", default="/data/ADNI_mariana/Parcellations_down/")
    parser.add_argument("--img-path", dest="img_folder", default="/data/ADNI_mariana/T1_down/")
    parser.add_argument("--atr-is-list", dest="atr_list", default=False)
    parser.add_argument("--batch-n", dest="batch_size", default=1)
    parser.add_argument("--gpu", dest="gpu", default=0)
    parser.add_argument("--amp", dest="amp", default=False)
    parser.add_argument("--load-model", dest="load_model", default="model_bio199.pt")
    parser.add_argument("--save-folder", dest="save_folder", default="./generated_imgs")

    args = parser.parse_args()
    return args


def main():
    opts = get_args_from_command_line()

    device = torch.device("cuda:" + str(opts.gpu) if torch.cuda.is_available() else "cpu")

    ds_inference = DataSetEval(
        region_list_path=opts.region_list_path,
        atr_folder=opts.atr_folder,
        seg_folder=opts.seg_folder,
        parc_folder=opts.parc_folder,
        atr_list=opts.atr_list,
    )
    inference_loader = torch.utils.data.DataLoader(
        ds_inference,
        batch_size=opts.batch_size,
        shuffle=opts.shuffle,
        pin_memory=opts.pin_memory,
        num_workers=opts.num_workers,
    )

    inference_transforms = get_transforms(mode="val")

    assert os.path.exists(opts.load_model), "Model file not found"
    model = torch.load(opts.load_model, map_location=device)

    inference_cycle(opts, model, inference_loader, inference_transforms, device)

    return


if __name__ == "__main__":
    main()
