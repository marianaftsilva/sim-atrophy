import torch
from training.losses import boundary_constraint_3d, W_neo_brain_3d, center_constraint_3d
from tqdm import tqdm
import torchvision
import nibabel as nib
import numpy as np
import os
from models import SpatialTransform

def train_cycle(opts, model, data_loader, transforms, optimizer, device):
    model.train()
    loss_epoch = 0
    loss1_epoch = 0
    loss2_epoch = 0
    loss3_epoch = 0
    with torch.amp.autocast(device=device, enabled=opts.amp), tqdm(data_loader, unit="batch") as tepoch:
        for input_seg, input_atrophy, center, _ in tepoch:
            input_atrophy = input_atrophy.to(device)
            input_seg = input_seg.to(device)
            center = center.to(device)

            # TODO: check if the transforms are applied correctly and if the input is in the correct value
            # input_seg = torchvision.tv_tensors.Mask(input_seg)
            # input_atrophy, input_seg = transforms(input_atrophy, input_seg)

            # TODO: remove this hard coding
            miu = torch.ones_like(input_seg)
            # miu = 0.01 for CSF
            miu[input_seg == 1] = 0.01
            # miu = 0 for background
            miu[input_seg == 0] = 0
            miu = miu.to(device)

            deformation = model(input_atrophy)

            # TODO: add if statement for dimensionality
            loss1 = W_neo_brain_3d(deformation, input_atrophy, miu, device) / torch.numel(input_seg)
            loss2 = boundary_constraint_3d(deformation, input_seg) / torch.numel(input_seg)
            loss3 = center_constraint_3d(deformation, center)

            total_loss = (loss1 * opts.w1) + (loss2 * opts.w2) + (loss3 * opts.w3)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            loss_epoch += total_loss

            loss1_epoch += loss1
            loss2_epoch += loss2
            loss3_epoch += loss3
            tepoch.set_postfix(loss=total_loss.item(), loss1=loss1.item(), loss2=loss2.item(), loss3=loss3.item())

    loss_epoch /= len(data_loader)
    loss1_epoch /= len(data_loader)
    loss2_epoch /= len(data_loader)
    loss3_epoch /= len(data_loader)

    return loss_epoch, loss1_epoch, loss2_epoch, loss3_epoch


def eval_cycle(opts, model, data_loader, transforms, device):
    model.eval()
    loss_epoch = 0
    loss1_epoch = 0
    loss2_epoch = 0
    loss3_epoch = 0
    with (
        torch.no_grad(),
        torch.amp.autocast(device=device, enabled=opts.amp),
        tqdm(data_loader, unit="batch") as tepoch,
    ):
        for input_seg, input_atrophy, center, _ in tepoch:
            input_atrophy = input_atrophy.to(device)
            input_seg = input_seg.to(device)
            center = center.to(device)

            # TODO: check if the transforms are applied correctly and if the input is in the correct value
            # input_seg = torchvision.tv_tensors.Mask(input_seg)
            # input_atrophy, input_seg = transforms(input_atrophy, input_seg)

            # TODO: remove this hard coding
            miu = torch.ones_like(input_seg)
            miu[input_seg == 1] = 0.01  # miu = 0.01 for CSF
            miu[input_seg == 0] = 0  # miu = 0 for background
            miu = miu.to(device)

            deformation = model(input_atrophy)

            # TODO: add if statement for dimensionality
            loss1 = W_neo_brain_3d(deformation, input_atrophy, miu, device) / torch.numel(input_seg)
            loss2 = boundary_constraint_3d(deformation, input_seg) / torch.numel(input_seg)
            loss3 = center_constraint_3d(deformation, center)

            total_loss = (loss1 * opts.w1) + (loss2 * opts.w2) + (loss3 * opts.w3)

            loss_epoch += total_loss
            loss1_epoch += loss1
            loss2_epoch += loss2
            loss3_epoch += loss3

            tepoch.set_postfix(loss=total_loss.item(), loss1=loss1.item(), loss2=loss2.item(), loss3=loss3.item())

    loss_epoch /= len(data_loader)
    loss1_epoch /= len(data_loader)
    loss2_epoch /= len(data_loader)
    loss3_epoch /= len(data_loader)

    return loss_epoch, loss1_epoch, loss2_epoch, loss3_epoch


def inference_cycle(opts, model, data_loader, transforms, device):
    model.eval()
    with (
        torch.no_grad(),
        torch.amp.autocast(device=device, enabled=opts.amp),
        tqdm(data_loader, unit="batch") as tepoch,
    ):
        for input_image, input_atrophy, affine, sub_id, original_size in tepoch:
            affine = affine.squeezed(0)
            if input_atrophy.max() > 2:
                break
            
            
            deformation = model(input_atrophy.to(device))

            # ATTENTION!!!
            # this is needed to fix a mismatch between the order of the axis defined in the biomechanical model
            # and the axis on the image. TO BE FIXED IN A FUTURE RELEASE
            new_flow = torch.ones_like(deformation)
            new_flow[:, 0, :, :, :] = deformation[:, 2, :, :, :]
            new_flow[:, 1, :, :, :] = deformation[:, 0, :, :, :]
            new_flow[:, 2, :, :, :] = deformation[:, 1, :, :, :]

            spatial_transform = SpatialTransform([input_image.shape[2], input_image.shape[3], input_image.shape[4]]).to(
                device
            )

            deformed_image = spatial_transform(input_image.to(device), new_flow)

            # save deformed image and flow with original image size
            final_image = np.zeros((182, 218, 182))
            final_image[2:178, 4:212, 2:178] = deformed_image.detach().cpu().numpy().squeeze(0).squeeze(0)

            flow_final = np.zeros((182, 218, 182, 3))
            flow_final[2:178, 4:212, 2:178, :] = new_flow.permute(0, 2, 3, 4, 1).detach().cpu().numpy().squeeze(0)

            # get det(F), which corresponds to computed atrophy
            F = gradient(deformation, device)
            F[:, :, :, :, 0, 0] = F[:, :, :, :, 0, 0] + 1
            F[:, :, :, :, 1, 1] = F[:, :, :, :, 1, 1] + 1
            F[:, :, :, :, 2, 2] = F[:, :, :, :, 2, 2] + 1
            det = torch.det(F).detach().cpu().numpy().squeeze(0)

            id_name = sub_id[0].split(".")[0].split("_to")[0]
            
            nib.save(
                nib.Nifti1Image(final_image, affine=affine),
                os.path.join(opts.save_folder, f"{id_name}_deformed_img.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(flow_final, affine=affine),
                os.path.join(opts.save_folder, f"{id_name}_deformation.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(det, affine=affine),
                os.path.join(opts.save_folder, f"{id_name}_deformation_estimated_atrophy.nii.gz"),
            )
