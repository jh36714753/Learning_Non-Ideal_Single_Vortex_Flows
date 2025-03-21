import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import LambdaLR, StepLR
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda")
real = torch.float32

L2_Loss = nn.MSELoss().cuda()

dropout_rate = 0
power = 1
eps_blob = 1e-10
scale = 1/(2*np.pi)

class Square(nn.Module):
    def forward(self, x):
        return x**2

class SineResidualBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # add shortcut
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
            )

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        out += self.shortcut(input)
        out = nn.functional.relu(out)
        return out

class Dynamics_Net(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 1
        out_dim = 1
        width = 40
        self.layers = nn.Sequential(SineResidualBlock(in_dim, width, omega_0=1., is_first=True),
                                SineResidualBlock(width, width, omega_0=1.),
                                SineResidualBlock(width, width, omega_0=1.),
                                SineResidualBlock(width, width, omega_0=1.),
                                nn.Linear(width, out_dim),
                                nn.LeakyReLU(negative_slope=-0.01)
                                )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class Position_Net(nn.Module):
    def __init__(self, num_vorts):
        super().__init__()
        in_dim = 1
        out_dim = num_vorts * 2
        self.layers = nn.Sequential(SineResidualBlock(in_dim, 64, omega_0=1., is_first=True),
                                SineResidualBlock(64, 128, omega_0=1.),
                                SineResidualBlock(128, 256, omega_0=1.),
                                SineResidualBlock(256, 512, omega_0=1.),
                                nn.Linear(512, out_dim)
                                )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
    
class w_Net(nn.Module):
    def __init__(self, num_vorts):
        super().__init__()
        in_dim = 1
        out_dim = num_vorts * 1
        self.layers = nn.Sequential(SineResidualBlock(in_dim, 64, omega_0=1., is_first=True),
                                SineResidualBlock(64, 128, omega_0=1.),
                                SineResidualBlock(128, 256, omega_0=1.),
                                SineResidualBlock(256, 512, omega_0=1.),
                                #nn.Linear(256, out_dim)
                                nn.Linear(512, out_dim)
                                )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
    
class size_square_Net(nn.Module):
    def __init__(self, num_vorts):
        super().__init__()
        in_dim = 1
        out_dim = num_vorts * 1
        self.layers = nn.Sequential(SineResidualBlock(in_dim, 64, omega_0=1., is_first=True),
                                SineResidualBlock(64, 128, omega_0=1.),
                                SineResidualBlock(128, 256, omega_0=1.),
                                SineResidualBlock(256, 512, omega_0=1.),
                                nn.Linear(512, out_dim),
                                nn.LeakyReLU(negative_slope=-0.01)
                                )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


def create_bundle(logdir, num_FP, num_VP, decay_step, decay_gamma, pretrain_dir = None):
    model_len = Dynamics_Net().to(device)
    model_FP_pos = Position_Net(num_FP).to(device)
    model_VP_pos = Position_Net(num_VP).to(device)
    w_pred = w_Net(num_VP).to(device)
    size_square_pred = size_square_Net(num_VP).to(device)
    grad_vars = list(model_len.parameters())
    grad_vars2 = list(model_FP_pos.parameters())
    grad_vars3 = list(model_VP_pos.parameters())
    grad_vars4 = list(w_pred.parameters())
    grad_vars5 = list(size_square_pred.parameters())
    ##########################
    # Load checkpoints
    ckpts = [os.path.join(logdir, f) for f in sorted(os.listdir(logdir)) if 'tar' in f]
    pretrain_ckpts = []
    if pretrain_dir:
        pretrain_ckpts = [os.path.join(pretrain_dir, f) for f in sorted(os.listdir(pretrain_dir)) if 'tar' in f]

    if len(ckpts) <= 0: # no checkpoints to load
        w_pred.requires_grad = True
        
        size_square_pred.requires_grad = True
        start = 0
        
        optimizer = torch.optim.AdamW([
            {'params': grad_vars, 'lr': 1e-3, 'weight_decay': 0.01},
            {'params': grad_vars2, 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': grad_vars3, 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': grad_vars4, 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': grad_vars5, 'lr': 1e-4, 'weight_decay': 0.01}
        ], lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
        
        # Load pretrained if there is one and no checkpoint exists
        if len(pretrain_ckpts) > 0:
            
            pre_ckpt_path = pretrain_ckpts[0]
            print ("[Initialize] Has pretrained available, reloading from: ", pre_ckpt_path)
            pre_ckpt = torch.load(pre_ckpt_path)
            print(pre_ckpt.keys())
            model_FP_pos.load_state_dict(pre_ckpt['model_FP_pos_state_dict'])
            
            pre_ckpt_path = pretrain_ckpts[1]
            print ("[Initialize] Has pretrained available, reloading from: ", pre_ckpt_path)
            pre_ckpt = torch.load(pre_ckpt_path)
            print(pre_ckpt.keys())
            model_VP_pos.load_state_dict(pre_ckpt['model_VP_pos_state_dict'])
            
            pre_ckpt_path = pretrain_ckpts[2]
            print ("[Initialize] Has pretrained available, reloading from: ", pre_ckpt_path)
            pre_ckpt = torch.load(pre_ckpt_path)
            print(pre_ckpt.keys())
            model_len.load_state_dict(pre_ckpt['model_len_state_dict'])
            
            pre_ckpt_path = pretrain_ckpts[3]
            print ("[Initialize] Has pretrained available, reloading from: ", pre_ckpt_path)
            pre_ckpt = torch.load(pre_ckpt_path)
            print(pre_ckpt.keys())
            size_square_pred.load_state_dict(pre_ckpt['size_square_pred_state_dict'])
            
            pre_ckpt_path = pretrain_ckpts[4]
            print ("[Initialize] Has pretrained available, reloading from: ", pre_ckpt_path)
            pre_ckpt = torch.load(pre_ckpt_path)
            print(pre_ckpt.keys())
            w_pred.load_state_dict(pre_ckpt['w_pred_state_dict'])

    else: # has checkpoints to load:
        ckpt_path = ckpts[-1]
        print ("[Initialize] Has checkpoint available, reloading from: ", ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        model_len.load_state_dict(ckpt["model_len_state_dict"])
        model_FP_pos.load_state_dict(ckpt["model_FP_pos_state_dict"])
        model_VP_pos.load_state_dict(ckpt["model_VP_pos_state_dict"])
        w_pred.load_state_dict(ckpt["w_pred_state_dict"])
        size_square_pred.load_state_dict(ckpt["size_square_pred_state_dict"])

        optimizer = torch.optim.AdamW([
            {'params': grad_vars, 'lr': 1e-3, 'weight_decay': 0.01},
            {'params': grad_vars2, 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': grad_vars3, 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': grad_vars4, 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': grad_vars5, 'lr': 1e-4, 'weight_decay': 0.01}
        ], lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100)

    net_dict = {
        'model_len' : model_len,
        'model_FP_pos' : model_FP_pos,
        'model_VP_pos' : model_VP_pos,
        'w_pred' : w_pred,
        'size_square_pred' : size_square_pred,
    }

    return net_dict, start, grad_vars, optimizer, lr_scheduler

# vels: [batch, width, height, 2]
def calc_div(vels):
    batch_size, width, height, D = vels.shape
    dx = 1./height
    
    du_dx = torch.zeros_like(vels[..., 0])
    dv_dy = torch.zeros_like(vels[..., 0])
    
    du_dx[:, 1:-1, 1:-1] = 1./(2*dx) * (vels[:, 2:, 1:-1, 0] - vels[:, :-2, 1:-1, 0])
    dv_dy[:, 1:-1, 1:-1] = 1./(2*dx) * (vels[:, 1:-1, 2:, 1] - vels[:, 1:-1, :-2, 1])
    
    du_dx[:, 0, :] = 1./dx * (vels[:, 1, :, 0] - vels[:, 0, :, 0])
    du_dx[:, -1, :] = 1./dx * (vels[:, -1, :, 0] - vels[:, -2, :, 0])
    
    dv_dy[:, :, 0] = 1./dx * (vels[:, :, 1, 1] - vels[:, :, 0, 1])
    dv_dy[:, :, -1] = 1./dx * (vels[:, :, -1, 1] - vels[:, :, -2, 1])
    
    return du_dx + dv_dy

# field: [batch, width, height, 1]
def calc_grad(field):
    batch_size, width, height, _ = field.shape
    dx = 1./height
    df_dx = 1./(2*dx) * (field[:, 2:, 1:-1] - field[:, :-2, 1:-1])
    df_dy = 1./(2*dx) * (field[:, 1:-1, 2:] - field[:, 1:-1, :-2])
    return torch.cat((df_dx, df_dy), dim = -1)

def calc_vort(vel_img, boundary = None): # compute the curl of velocity
    B, W, H, _ = vel_img.shape
    dx = 1./H
    vort_img = torch.zeros(B, W, H, 1, device = device, dtype = real)
    u = vel_img[...,[0]]
    v = vel_img[...,[1]]
    dvdx = 1/(2*dx) * (v[:, 2:, 1:-1] - v[:, :-2, 1:-1])
    dudy = 1/(2*dx) * (u[:, 1:-1, 2:] - u[:, 1:-1, :-2])
    vort_img[:, 1:-1, 1:-1] = dvdx - dudy
    if boundary is not None:
        # set out-of-bound pixels to 0 because velocity undefined there
        OUT = (boundary[0] >= -boundary[2] - 4)
        OUT = OUT.unsqueeze(0).repeat(B, 1, 1).unsqueeze(-1)
        vort_img[OUT] *= 0
    return vort_img

import torch

def calc_laplacian(field):
    """
    Compute the Laplacian of a field (∇²ω) using zero-gradient boundary conditions
    
    Parameters:
    field: torch.Tensor, vorticity field with shape [batch, width, height]
    
    Returns:
    torch.Tensor, the Laplacian ∇²ω with shape [batch, width, height]
    """
    batch_size, width, height = field.shape
    dx = 1./height  

    padded_field = torch.zeros(batch_size, width+2, height+2, device=field.device)
    padded_field[:, 1:-1, 1:-1] = field

    padded_field[:, 0, 1:-1] = field[:, 0, :]  
    padded_field[:, -1, 1:-1] = field[:, -1, :]  
    padded_field[:, 1:-1, 0] = field[:, :, 0]  
    padded_field[:, 1:-1, -1] = field[:, :, -1]  
    
    padded_field[:, 0, 0] = field[:, 0, 0]
    padded_field[:, 0, -1] = field[:, 0, -1]
    padded_field[:, -1, 0] = field[:, -1, 0]
    padded_field[:, -1, -1] = field[:, -1, -1]

    d2f_dx2 = (padded_field[:, 2:, 1:-1] - 2*padded_field[:, 1:-1, 1:-1] + padded_field[:, :-2, 1:-1]) / (dx**2)

    d2f_dy2 = (padded_field[:, 1:-1, 2:] - 2*padded_field[:, 1:-1, 1:-1] + padded_field[:, 1:-1, :-2]) / (dx**2)

    laplacian = d2f_dx2 + d2f_dy2

    return laplacian

def bilinear_interpolate(field, coords):
    """
    Perform bilinear interpolation on the field at given coordinates.
    
    Args:
    field (torch.Tensor): Tensor of shape [batch, width, height, channels]
    coords (torch.Tensor): Tensor of shape [batch, num_points, 2] containing x, y coordinates in range [0, 1]
    
    Returns:
    torch.Tensor: Interpolated values of shape [batch, num_points, channels]
    """
    batch, width, height, channels = field.shape
    
    # Scale coordinates to grid indices
    x = coords[..., 0] * (width - 1)
    y = coords[..., 1] * (height - 1)
    
    # Compute the four nearest grid points
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    
    # Clip to ensure we're within bounds
    x0 = torch.clamp(x0, 0, width - 1)
    x1 = torch.clamp(x1, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    y1 = torch.clamp(y1, 0, height - 1)
    
    # Compute weights
    wx = (x - x0.float()).unsqueeze(-1)
    wy = (y - y0.float()).unsqueeze(-1)
    
    # Perform bilinear interpolation
    v00 = field[torch.arange(batch).unsqueeze(1), x0, y0]
    v01 = field[torch.arange(batch).unsqueeze(1), x0, y1]
    v10 = field[torch.arange(batch).unsqueeze(1), x1, y0]
    v11 = field[torch.arange(batch).unsqueeze(1), x1, y1]
    
    interp = (v00 * (1 - wx) * (1 - wy) +
              v01 * (1 - wx) * wy +
              v10 * wx * (1 - wy) +
              v11 * wx * wy)
    
    return interp

# sdf: [W, H]
# sdf normal: [W, H, 2]
def calc_sdf_normal(sdf):
    W, H = sdf.shape
    sdf_normal = torch.zeros((W, H, 2)).cuda() #[W, H, 2]
    sdf_normal[1:-1, 1:-1] = calc_grad(sdf[None,...,None])[0] # outward pointing [W, H, 2]
    sdf_normal = F.normalize(sdf_normal, dim = -1, p = 2)
    return sdf_normal

# vorts_pos: [batch, num_vorts, 2] 
# query_pos: [num_query, 2] or [batch, num_query, 2]
# return: [batch, num_queries, num_vorts, 2]
def calc_diff_batched(_vorts_pos, _query_pos):
    vorts_pos = _vorts_pos[:, None, :, :] # [batch, 1, num_vorts, 2]
    if len(_query_pos.shape) > 2:
        query_pos = _query_pos[:, :, None, :] # [batch, num_query, 1, 2]
    else: 
        query_pos = _query_pos[None, :, None, :] # [1, num_query, 1, 2]
    diff = query_pos - vorts_pos # [batch, num_queries, num_vorts, 2]
    return diff


# vorts_pos shape: [batch, num_vorts, 2]
# vorts_w shape: [num_vorts, 1] or [batch, num_vorts, 1]
# vorts_size shape: [num_vorts, 1] or [batch, num_vorts, 1]
def vort_to_vel(network_length, vorts_size, vorts_w, vorts_pos, query_pos, length_scale):
    diff = calc_diff_batched(vorts_pos, query_pos) # [batch_size, num_query, num_query, 2]
    # some broadcasting
    if len(vorts_size.shape) > 2:
        blob_size = vorts_size[:, None, ...] # [batch, 1, num_vorts, 1]
    else:
        blob_size = vorts_size[None, None, ...] # [1, 1, num_vorts, 1] 
    if len(vorts_w.shape) > 2:
        vorts_w = vorts_w[:, None, ...] # [batch, num_query, num_vort, 1]
    else:
        vorts_w = vorts_w[None, None, ...] # [1, 1, num_vort, 1]
    
    diff = calc_diff_batched(vorts_pos, query_pos)
    dist = torch.norm(diff, dim = -1, p = 2, keepdim = True) 

    # cross product in 2D
    R = diff.flip([-1]) # (x, y) becomes (y, x)
    R[..., 0] *= -1 # (y, x) becomes (-y, x)
    R = F.normalize(R, dim = -1)
    
    blob_size = blob_size * scale 
    
    dist = dist / (blob_size/length_scale)
    dist = torch.pow(dist, power) 
    magnitude = network_length(dist)

    result = magnitude * R * vorts_w
    result = torch.sum(result, dim = -2) # [batch_size, num_queries, 2]

    return result