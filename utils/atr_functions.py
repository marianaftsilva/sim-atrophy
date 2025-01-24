#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:45:39 2021

@author: mds19
"""
import nibabel as nib
import numpy as np
import pandas as pd

def atr_list_to_vol(atr_list, atr_name, seg_folder):
    region_list = np.load('region_list.npy')
    
    seg1 = nib.load(seg_folder + atr_name.split('_to')[0] + '.nii.gz')
    seg1 = np.asanyarray(seg1.dataobj)
    
    atr_vol = np.ones_like(seg1).astype(np.float32)
    
    for i in range(0,138):
        atr_vol[seg1 == region_list[i]] = atr_list[i] 
    
    return atr_vol

def atr_list_to_vol28(atr_list, atr_name, seg_folder):
    
    seg1 = nib.load(seg_folder + atr_name.split('_to')[0] + '.nii.gz')
    seg1 = np.asanyarray(seg1.dataobj)
    
    atr_vol = np.ones_like(seg1).astype(np.float32)
    
    for i in range(0,28):
        atr_vol[seg1 == i] = atr_list[i] 
    
    return atr_vol

def create_rand_atr():
    
    region_list = np.load('region_list.npy')
    
    atr_max = pd.read_pickle('atr_max.pkl')
    atr_min = pd.read_pickle('atr_min.pkl')
    
    atr_list = [] 
    for i in range(0,138):
        atr_list.append(np.random.uniform(atr_min[region_list[i]]-0.2, atr_max[region_list[i]]+0.2))
    
    return atr_list
