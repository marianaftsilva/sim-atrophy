#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mds19
"""

# imports
import os
import numpy as np
import torch
import nibabel as nib
import torch.utils.data
import scipy.ndimage as ndi

# internal imports
from utils.atr_functions import atr_list_to_vol
from torchvision import transforms


class DataSetTrain(torch.utils.data.Dataset):
    def __init__(self, region_list_path, atr_folder, seg_folder, parc_folder=None, atr_list=False):
        self.region_list_path = region_list_path
        self.seg_folder = seg_folder
        self.atr_folder = atr_folder
        self.atr_paths = sorted(os.listdir(self.atr_folder))
        self.atr_list = atr_list
        self.parc_folder = parc_folder

    def __len__(self):
        return len(self.atr_paths)

    def __getitem__(self, idx):
        atr_name = sorted(self.atr_paths)[idx]

        # if using a list of atrophies, convert to volume based on the parcelations
        if self.atr_list:
            atr = np.load(self.atr_folder + atr_name)
            atr = atr_list_to_vol(self.region_list_path, atr, atr_name, self.parc_folder)

        else:
            atr = nib.load(self.atr_folder + atr_name)

        atr = atr.astype(np.float32)[5:85, 6:102, 5:85]
        atr = torch.from_numpy(atr).unsqueeze(0)

        # create a range of atrophy values
        patr = atr
        ran = -1.5 * torch.rand(1) + 2
        patr[atr > 1] *= ran
        patr[atr < 1] /= ran

        # get segmentation
        seg = nib.load(self.seg_folder + atr_name.split(".")[0].split("_to")[0] + ".nii.gz")
        seg = np.asanyarray(seg.dataobj)
        seg = seg.astype(np.float32)[5:85, 6:102, 5:85]

        # get coordinates of center of mass
        c1, c2, c3 = ndi.center_of_mass(seg)
        seg = torch.from_numpy(seg).unsqueeze(0)

        # create binary mask of center
        center = torch.zeros_like(seg)
        center[:, np.int(c1), np.int(c2), np.int(c3)] = 1

        return seg, patr, center, atr_name


class DataSetEval(torch.utils.data.Dataset):
    def __init__(self, region_list_path, atr_folder, img_folder, parc_folder=None, atr_list=False):
        self.region_list_path = region_list_path
        self.img_folder = img_folder
        self.atr_folder = atr_folder
        self.atr_paths = sorted(os.listdir(self.atr_folder))
        self.atr_list = atr_list
        self.parc_folder = parc_folder

    def __len__(self):
        return len(self.atr_paths)

    def __getitem__(self, idx):
        atr_name = sorted(self.atr_paths)[idx]

        # if using a list of atrophies, convert to volume based on the parcelations
        if self.atr_list:
            atr = np.load(self.atr_folder + atr_name)
            atr = atr_list_to_vol(self.region_list_path, atr, atr_name, self.parc_folder)
        else:
            atr = nib.load(self.atr_folder + atr_name)

        atr = atr.astype(np.float32)[5:85, 6:102, 5:85]
        atr = torch.from_numpy(atr).unsqueeze(0)

        # get image
        img = nib.load(self.img_folder + atr_name)
        # get affine
        affine = img.affine
        img = np.asanyarray(img.dataobj)
        org_size = img.shape
        img = img.astype(np.float32)[5:85, 6:102, 5:85]
        img = torch.from_numpy(img).unsqueeze(0)

        return img, atr, affine, atr_name, org_size


class AtrophyDataSet(torch.utils.data.Dataset):
    def __init__(self, region_list_path, atr_folder, img_folder, parc_folder=None, atr_list=False, mode="train"):
        self.region_list_path = region_list_path
        self.img_folder = img_folder
        self.atr_folder = atr_folder
        self.atr_paths = sorted(os.listdir(self.atr_folder))
        self.atr_list = atr_list
        self.parc_folder = parc_folder
        self.mode = mode

    def __len__(self):
        return len(self.atr_paths)

    def __getitem__(self, idx):
        atr_name = sorted(self.atr_paths)[idx]

        # if using a list of atrophies, convert to volume based on the parcelations
        if self.atr_list:
            atr = np.load(self.atr_folder + atr_name)
            atr = atr_list_to_vol(self.region_list_path, atr, atr_name, self.parc_folder)
        else:
            atr = nib.load(self.atr_folder + atr_name)

        atr = atr.astype(np.float32)[5:85, 6:102, 5:85]
        atr = torch.from_numpy(atr).unsqueeze(0)

        # get image
        img = nib.load(self.img_folder + atr_name)
        # get affine
        affine = img.affine
        img = np.asanyarray(img.dataobj)
        org_size = img.shape
        img = img.astype(np.float32)[5:85, 6:102, 5:85]
        img = torch.from_numpy(img).unsqueeze(0)

        return img, atr, affine, atr_name, org_size


def get_transforms(mode="train"):
    """
    Define transforms for data augmentation
    """
    if mode == "train":
        # Training transforms with augmentation
        transforms_out = transforms.Compose(
            [
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    elif mode == "val":
        transforms_out = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])

    return transforms_out
