# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:44:39 2020

@author: Mariana
"""

import torch
import torch.nn.functional as nnF

def gradient(u,device):    
    
    weights_y = torch.tensor([[0., 0., 0.],
                        [-0.5, 0., 0.5],
                        [0., 0., 0.]])
    
    weights_y = weights_y.view(1, 1, 3, 3).repeat(2, 1, 1, 1).to(device)

    
    weights_x = torch.tensor([[0.,-0.5, 0.],
                        [0., 0., 0.],
                        [0., 0.5, 0.]])
    
    weights_x = weights_x.view(1, 1, 3, 3).repeat(2, 1, 1, 1).to(device)
    
    du_dx = nnF.conv2d(u, weights_x, padding = 1, groups = 2)
    du_dy= nnF.conv2d(u, weights_y, padding = 1, groups = 2)
    
    du_dx = du_dx.permute(0,2,3,1).unsqueeze(4) # 1xHxWx2x1
    du_dy = du_dy.permute(0,2,3,1).unsqueeze(4) # 1xHxWx2x1

    Grad = torch.cat((du_dx,du_dy), dim = 4) # 1xHxWx2x2
    
    return Grad

def trace(M):
    
    trace = torch.add(M[:,:,:,0,0], M[:,:,:,1,1])
    
    return trace.unsqueeze(3)

    
def create_inv_growth_tensor(a,device):
    
    G_inv = torch.zeros((a.shape[0],a.shape[1],a.shape[2],2,2)).to(device)
    a = a.squeeze(3) + 1e-5
    
    G_inv[:,:,:,0,0] = a**(-1/2)
    G_inv[:,:,:,1,1] = a**(-1/2)
    
    return G_inv
        
def W_neo_brain(u, a, miu,device):

    miu = miu.permute(0,2,3,1)
    a = a.permute(0,2,3,1)

    F = gradient(u,device) 
    F[:,:,:,0,0] = F[:,:,:,0,0] + 1
    F[:,:,:,1,1] = F[:,:,:,1,1] + 1
   
    G_inv = create_inv_growth_tensor(a,device)
    K = torch.matmul(F,G_inv)
    
    J = torch.det(K).unsqueeze(3)
    trace_1 = trace(torch.matmul(K,torch.transpose(K,3,4)))
       
    W = 50*miu*((J-1)**2) + (0.5*miu*((trace_1*(J**(-1)))-2))
    
    W = W.permute(0,2,3,1)
    
    return torch.sum(W**2)


def atrophy_constraint(u, a, p = 1):
    u = u.permute(0,2,3,1)
    
    a = a.permute(0,2,3,1)
    
    p = torch.zeros_like(a)
    p[a!=0] = 1
    
    at_cons = torch.mean((p*(trace(gradient(u)) + a))**2)
    
    return at_cons 

def boundary_constraint(u, img):
    

    u = u.permute(0,2,3,1)


    img = img.permute(0,2,3,1)
     
    ux2_uy2 = torch.sum(u**2,dim = 3, keepdim= True)

    boundary = torch.sum((ux2_uy2[img==0]))
    
    return boundary

# def center_constraint(u, c):
    
#     u = u.permute(0,2,3,1)
#     c = c.permute(0,2,3,1)
    
#     u_x = c*(u[:,:,:,0].unsqueeze(3))
#     u_y = c*(u[:,:,:,1].unsqueeze(3))
    
#     center =  (torch.sum(u_x**2) + torch.sum(u_y**2))
    
#     return center

def flow_constraint(u,img):
    u = u.permute(0,2,3,1)
    img = img.permute(0,2,3,1)
    
    p = torch.zeros_like(img)
    p[img>0] = 1
   
    u_x = p*(u[:,:,:,0].unsqueeze(3))
    u_y = p*(u[:,:,:,1].unsqueeze(3))

    u_sum = (torch.sum(u_x)**2 + torch.sum(u_y)**2)
             
    return u_sum


    
