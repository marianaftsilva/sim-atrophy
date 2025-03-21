#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mds19
"""

# imports
import torch
from torch.optim import Adam
import torch.utils.data
from optparse import OptionParser
from utils.dataloader import DataSetTrain

# internal imports
from models.network_3d import unet_core
from training.losses import W_neo_brain, boundary_constraint  # , center_constraint

def main():
    # define args
    parser = OptionParser()
    parser.add_option("--atr_path", dest="atr_folder", default="Data/atr_folder/")
    parser.add_option("--seg_path", dest="seg_folder", default="Data/seg_folder/")
    # parser.add_option("--parc_path", dest="parc_folder", default='/data/ADNI_mariana/Parcellations_down/')
    # parser.add_option("--atr_is_list", dest="atr_list", default=True)
    parser.add_option("--batch_n", dest="batch_size", default=10)
    parser.add_option("--gpu", dest="gpu", default=0)
    parser.add_option("--nf_enc", dest="nf_enc", default=[16, 32, 32, 32])
    parser.add_option("--nf_dec", dest="nf_dec", default=[32, 32, 32, 32, 8, 8])
    parser.add_option("--epochs", dest="epochs", default=200)
    parser.add_option("--n_save_iter", dest="n_save_iter", default=1)
    parser.add_option("--lr", dest="lr", default=1e-5)
    parser.add_option("--w_losses", dest="weight", default=10)
    parser.add_option("--load_model", dest="load_model", default=None)

    (opts, _) = parser.parse_args()

    device = torch.device("cuda:" + str(opts.gpu) if torch.cuda.is_available() else "cpu")
    print(device)

    # load data
    ds_train = DataSetTrain(atr_folder=opts.atr_folder, seg_folder=opts.seg_folder)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=opts.batch_size, shuffle=True)

    # load model
    if opts.load_model is None:
        model = unet_core(dim=2, enc_df=opts.nf_enc, dec_nf=opts.nf_dec, n_in=1).to(device)
    else:
        model = torch.load(opts.load_model, map_location=device)
    
    optimizer = Adam(model.parameters(), lr=opts.lr)

    for i in range(0, opts.epochs):
        model.train()
        loss_epoch = 0
        loss1_epoch = 0
        loss2_epoch = 0
        for k, (input_seg, input_atrophy, _) in enumerate(train_loader, 0):

            input_atrophy = input_atrophy.to(device).requires_grad_()
            input_seg = input_seg.to(device).requires_grad_()

            miu = input_seg.to(device).requires_grad_()

            deformation = model(input_atrophy)

            loss1 = W_neo_brain(deformation, input_atrophy, miu, device) / torch.numel(input_seg)
            loss2 = (boundary_constraint(deformation, input_seg) / torch.numel(input_seg)) * opts.weight / 10

            total_loss = loss1 + loss2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_epoch += total_loss

            loss1_epoch += loss1
            loss2_epoch += loss2

        if i % opts.n_save_iter == 0:
            print(i, loss_epoch.item() / k, loss1_epoch.item() / k, loss2_epoch.item() / k)

            torch.save(model, "model_bio" + str(i) + ".pt")

    return


if __name__ == "__main__":
    main()
