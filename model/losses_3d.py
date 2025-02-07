# -*- coding: utf-8 -*-
"""
@author: Mariana
"""

import torch
import torch.nn.functional as nnF

def gradient(u, device):    

    weights_zero = torch.zeros((3,3,3))
    
    weights_y = weights_zero.clone()
    weights_y[1,1,0] = -0.5
    weights_y[1,1,2] = 0.5

    weights_y = weights_y.view(1, 1, 3, 3, 3).repeat(3, 1, 1, 1, 1)
    
    weights_x = weights_zero.clone()
    weights_x[1,0,1] = -0.5
    weights_x[1,2,1] = 0.5
    
    weights_x = weights_x.view(1, 1, 3, 3, 3).repeat(3, 1, 1, 1, 1)
    
    weights_z = weights_zero.clone()
    weights_z[0,1,1] = -0.5
    weights_z[2,1,1] = 0.5
    
    weights_z = weights_z.view(1, 1, 3, 3, 3).repeat(3, 1, 1, 1, 1)
    
    du_dx = nnF.conv3d(u, weights_x.to(device), padding = 1, groups = 3)
    du_dy = nnF.conv3d(u, weights_y.to(device),padding = 1, groups = 3)
    du_dz = nnF.conv3d(u, weights_z.to(device),padding = 1, groups = 3)
    
    du_dx = du_dx.permute(0,2,3,4,1).unsqueeze(5) # 1xHxWxDx3x1
    du_dy = du_dy.permute(0,2,3,4,1).unsqueeze(5)  # 1xHxWxDx3x1
    du_dz = du_dz.permute(0,2,3,4,1).unsqueeze(5)  # 1xHxWxDx3x1
 
    Grad = torch.cat((du_dx,du_dy,du_dz), dim = 5)# 1xHxWxDx3x3
    
    return Grad

def trace(M):
    
    trace = torch.add(torch.add(M[:,:,:,:,0,0], M[:,:,:,:,1,1]), M[:,:,:,:,2,2])
    
    return trace.unsqueeze(4)

def create_inv_growth_tensor(a, device):
    
    G_inv = torch.zeros((a.shape[0],a.shape[1],a.shape[2],a.shape[3],3,3)).to(device)
    a = a.squeeze(4) + 1e-5
    
    G_inv[:,:,:,:,0,0] = a**(-1/3)
    G_inv[:,:,:,:,1,1] = a**(-1/3)
    G_inv[:,:,:,:,2,2] = a**(-1/3)
    
    return G_inv
        
def W_neo_brain(u, a, miu, device):

    miu = miu.permute(0,2,3,4,1)
    a = a.permute(0,2,3,4,1)

    F = gradient(u, device) 
    F[:,:,:,:,0,0] = F[:,:,:,:,0,0] + 1
    F[:,:,:,:,1,1] = F[:,:,:,:,1,1] + 1
    F[:,:,:,:,2,2] = F[:,:,:,:,2,2] + 1
   
    G_inv = create_inv_growth_tensor(a, device)
    K = torch.matmul(F,G_inv)
    
    J = torch.det(K).unsqueeze(4)
    trace_1 = trace(torch.matmul(K,torch.transpose(K,4,5)))
       
    W = 50*miu*((J-1)**2) + (0.5*miu*(trace_1*(torch.abs(J)**(-2/3))-3))

    W = W.permute(0,4,1,2,3)
    
    return torch.sum(W**2)

def atrophy_constraint(u, a, p = 1):
    u = u.permute(0,2,3,4,1)
    
    a = a.permute(0,2,3,4,1)
    
    p = torch.zeros_like(a)
    p[a!=0] = 1
    
    at_cons = torch.mean((p*(trace(gradient(u)) + a))**2)
    
    return at_cons 

def boundary_constraint(u, img):

    u = u.permute(0,2,3,4,1)
    img = img.permute(0,2,3,4,1)
     
    ux2_uy2 = torch.sum(u**2,dim = 4, keepdim= True)
    
    boundary = torch.sum((ux2_uy2[img==0]))
    
    return boundary

def center_constraint(u, c):
    
    u = u.permute(0,2,3,4,1)
    c = c.permute(0,2,3,4,1)

    u_x = c*(u[:,:,:,:,0].unsqueeze(4))
    u_y = c*(u[:,:,:,:,1].unsqueeze(4))
    u_z = c*(u[:,:,:,:,2].unsqueeze(4))
    
    center =  (torch.sum(u_x**2) + torch.sum(u_y**2) + torch.sum(u_z**2)) 
    
    return center
