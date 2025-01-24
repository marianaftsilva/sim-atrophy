#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:41:34 2022

@author: mds19
"""

# imports
import torch
import torch.utils.data
from optparse import OptionParser
import nibabel as nib
import numpy as np

# internal imports 
from model.network_3d import SpatialTransform
from model.losses_3d import gradient
from utils.dataloader import DataSetEval

def main():
    
    # define args
    parser = OptionParser()
    parser.add_option("--atr_path", dest="atr_folder", default='/atr_folder/')
    parser.add_option("--img_path", dest="img_folder", default='/img_folder/')
    parser.add_option("--parc_path", dest="parc_folder", default='/parc_folder/')
    parser.add_option("--atr_is_list", dest="atr_list", default=False)
    parser.add_option("--batch_n", dest="batch_size", default=1)
    parser.add_option("--gpu", dest='gpu', default=0)
    parser.add_option("--load_model", dest='load_model', default='trained_model_3d.pt')
    
    (opts, args) = parser.parse_args()
    
    device = torch.device('cuda:' + str(opts.gpu) if torch.cuda.is_available() else 'cpu')
    
    # load data
    ds_test = DataSetEval(atr_folder = opts.atr_folder, img_folder = opts.img_folder, parc_folder = opts.parc_folder, atr_list = opts.atr_list)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size = opts.batch_size, shuffle = False) 
    
    model = torch.load(opts.load_model, map_location = device)
    
    model.eval()
    for k, (input_image, input_atrophy, affine, sub_id, original_size) in enumerate(test_loader, 0):

        deformation = model(input_atrophy.to(device))

        # ATTENTION!!!
        # this is needed to fix a mismatch between the order of the axis defined in the biomechanical model
        # and the axis on the image. TO BE FIXED IN A FUTURE RELEASE       
        new_flow = torch.ones_like(deformation)
        new_flow[:,0,:,:,:] = deformation[:,2,:,:,:]
        new_flow[:,1,:,:,:] = deformation[:,0,:,:,:]
        new_flow[:,2,:,:,:] = deformation[:,1,:,:,:]
        
        spatial_transform = SpatialTransform([input_image.shape[2],input_image.shape[3],input_image.shape[4]]).to(device)
        
        deformed_image = spatial_transform(input_image.to(device),new_flow)
        
        # save deformed image and flow with original image size
        final_image = np.zeros((182,218,182))
        final_image[2:178,4:212,2:178] = deformed_image.detach().cpu().numpy().squeeze(0).squeeze(0)
        
        flow_final = np.zeros((182,218,182,3))
        flow_final[2:178,4:212,2:178,:] = new_flow.permute(0,2,3,4,1).detach().cpu().numpy().squeeze(0)
        
        # get det(F), which corresponds to computed atrophy
        F = gradient(deformation, device) 
        F[:,:,:,:,0,0] = F[:,:,:,:,0,0] + 1
        F[:,:,:,:,1,1] = F[:,:,:,:,1,1] + 1
        F[:,:,:,:,2,2] = F[:,:,:,:,2,2] + 1
        det = torch.det(F).detach().cpu().numpy().squeeze(0)
        
        nib.save(nib.Nifti1Image(final_image, affine=affine.squeeze(0)), 'results/' + sub_id[0].split('.')[0].split('_to')[0] + '_deformed_img.nii.gz')  
        nib.save(nib.Nifti1Image(flow_final, affine=affine.squeeze(0)), 'results/' + sub_id[0].split('.')[0].split('_to')[0] + '_deformation.nii.gz')
        nib.save(nib.Nifti1Image(det, affine=affine.squeeze(0)), 'results/' + sub_id[0].split('.')[0].split('_to')[0] + '_estimated_atrophy.nii.gz')
        
    return

if __name__ == "__main__":
    main()