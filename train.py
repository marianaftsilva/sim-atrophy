#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mds19
"""

import torch
from torch.optim import Adam
import torch.utils.data
from utils.dataloader import DataSetTrain, DataSetEval, get_transforms

from models.network_3d import unet_core
from training.epoch_cycles import train_cycle, eval_cycle
import numpy as np
import os
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter


def get_args_from_command_line():
    parser = ArgumentParser(description="Parser for training Sim Atrophy")

    # Data Paths
    parser.add_argument("--region-list-path", dest="region_list_path", default="./region_list.npy")
    parser.add_argument("--atr-path", dest="atr_folder", default="/data/ADNI_mariana/Train/")
    parser.add_argument("--seg-path", dest="seg_folder", default="/data/ADNI_mariana/Seg_down/")
    parser.add_argument("--parc-path", dest="parc_folder", default="/data/ADNI_mariana/Parcellations_down/")
    parser.add_argument("--img-path", dest="img_folder", default="/data/ADNI_mariana/T1_down/")
    parser.add_argument("--save-folder", dest="save_folder", default="./results_folder")

    # TODO: remove this variable
    parser.add_argument("--atr-is-list", help="atri", dest="atr_list", default=True, action="store_true")

    # Model Parameters
    parser.add_argument("--n-dims", dest="n_dims", default=3, type=int)
    parser.add_argument("--nf-enc", dest="nf_enc", default=[16, 32, 32, 32], type=int, nargs="+")
    parser.add_argument("--nf-dec", dest="nf_dec", default=[32, 32, 32, 32, 8, 8], type=int, nargs="+")
    parser.add_argument("--load-model", dest="load_model", default=None)
    parser.add_argument("--compile-model", dest="compile_model", default=True)

    # Training Parameters
    parser.add_argument("--epochs", dest="epochs", default=200, type=int)
    parser.add_argument("--n-save-epoch", dest="n_save_epoch", default=1, type=int)
    parser.add_argument("--lr", dest="lr", default=1e-5, type=float)
    parser.add_argument("--loss-1-weight", dest="w1", default=1, type=float)
    parser.add_argument("--loss-2-weight", dest="w2", default=0.1, type=float)
    parser.add_argument("--loss-3-weight", dest="w3", default=100, type=float)
    parser.add_argument("--seed", dest="seed", default=42, type=int)
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        help="Bool to shuffle train and val sets (test set will not be shuffled in any situation)",
        default=True,
        action="store_true",
    )

    # Data Loader Parameters
    parser.add_argument("--batch-size", dest="batch_size", default=8, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--pin-memory", dest="pin_memory", default=True, action="store_true")
    parser.add_argument("--num-workers", dest="num_workers", default=4, type=int)

    args = parser.parse_args()
    return args


def main():
    opts = get_args_from_command_line()
    writer = SummaryWriter(log_dir=os.path.join(opts.save_folder, "logs"))

    if opts.seed is not None:
        torch.manual_seed(int(opts.seed))

    if opts.gpu is not None:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autograd.profiler.profile(enabled=False)

    device = torch.device("cuda:" + str(opts.gpu) if torch.cuda.is_available() else "cpu")

    train_transforms = get_transforms(mode="train")
    val_transforms = get_transforms(mode="val")

    # TODO: check how the train and eval datasets are created and make it one function, specifying the train/val/test mode
    ds_train = DataSetTrain(
        region_list_path=opts.region_list_path,
        atr_folder=opts.atr_folder,
        seg_folder=opts.seg_folder,
        parc_folder=opts.parc_folder,
        atr_list=opts.atr_list,
    )
    ds_val = DataSetEval(
        region_list_path=opts.region_list_path,
        atr_folder=opts.atr_folder,
        seg_folder=opts.seg_folder,
        parc_folder=opts.parc_folder,
        atr_list=opts.atr_list,
    )

    train_loader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=opts.batch_size,
        shuffle=opts.shuffle,
        pin_memory=opts.pin_memory,
        num_workers=opts.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        ds_val,
        batch_size=opts.batch_size,
        shuffle=opts.shuffle,
        pin_memory=opts.pin_memory,
        num_workers=opts.num_workers,
    )

    # TODO: write a description of input parameters, especially n_in
    model = unet_core(dim=opts.n_dim, enc_df=opts.nf_enc, dec_nf=opts.nf_dec, n_in=1)

    if opts.load_model is not None:
        model = torch.load(opts.load_model, map_location="cpu")

    model.to(device)

    if opts.compile_model:
        model = torch.compile(model)

    optimizer = Adam(model.parameters(), lr=opts.lr)

    os.makedirs(opts.save_folder, exist_ok=True)

    best_val_loss = np.inf
    for ii in range(opts.epochs):
        train_loss, train_loss1, train_loss2, train_loss3 = train_cycle(
            opts, model, train_loader, train_transforms, optimizer, device
        )
        val_loss, val_loss1, val_loss2, val_loss3 = eval_cycle(opts, model, val_loader, val_transforms, device)

        print(f"[Epoch {ii}] Total Train Loss: {train_loss.item()}, Total Validation Loss: {val_loss.item()}")

        # Log losses to tensorboard
        writer.add_scalar("Loss/train/total", train_loss.item(), ii)
        writer.add_scalar("Loss/train/loss1", train_loss1.item(), ii)
        writer.add_scalar("Loss/train/loss2", train_loss2.item(), ii)
        writer.add_scalar("Loss/train/loss3", train_loss3.item(), ii)

        writer.add_scalar("Loss/validation/total", val_loss.item(), ii)
        writer.add_scalar("Loss/validation/loss1", val_loss1.item(), ii)
        writer.add_scalar("Loss/validation/loss2", val_loss2.item(), ii)
        writer.add_scalar("Loss/validation/loss3", val_loss3.item(), ii)

        if ii % opts.n_save_epoch == 0:
            torch.save(model, os.path.join(opts.save_folder, f"model_bio_epoch_{str(ii)}.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, os.path.join(opts.save_folder, "model_bio_best.pt"))
            print(f"=====Best model saved at epoch {ii}=====")

    # Close tensorboard writer
    writer.close()
    return


if __name__ == "__main__":
    main()
