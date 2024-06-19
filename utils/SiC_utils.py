# This code belongs to the paper
#
# Sebastian Neumayer and Fabian Altekrüger (2024)
# Stability of Data-Dependent Ridge-Regularization for Inverse Problems
#
# Please cite the paper, if you use the code.

import math
import torch
from torch import nn
import skimage.io as io

def imread(img_name, device, as_gray=False):
    '''
    loads an image as torch.tensor on the selected device
    ''' 
    np_img = io.imread(img_name, as_gray=as_gray)
    tens_img = torch.tensor(np_img, dtype=torch.float, device=device)
    if torch.max(tens_img) > 1:
        tens_img/=255
    if len(tens_img.shape) < 3:
        tens_img = tens_img.unsqueeze(2)
    if tens_img.shape[2] > 3:
        tens_img = tens_img[:,:,:3]
    tens_img = tens_img.permute(2,0,1)
    return tens_img.unsqueeze(0)

class superresolution_operators(nn.Module):
    '''
    defines a superresolution operator, which is a strided convolution with 
    a gaussian blur kernel
    ''' 
    def __init__(self, factor, device):
        super().__init__()
        self.gaussian_std = factor/2
        self.factor = factor
        self.device = device
        self.kernel_size = 16
        self.gaussian = gaussian_downsample(kernel_size=self.kernel_size, 
                            sigma=self.gaussian_std,stride=self.factor,
                            device=device)
        
    def apply_forward(self,x):
        return self.gaussian(x)
    
    def apply_adjoint(self,x):
        return self.gaussian.transposed(x)
        
    def get_op_norm(self, img_size, n_iter=15):
        '''
        computes the operator norm for a given operator
        '''
        x = torch.rand((1, 1, img_size[0],img_size[1])).to(self.device).requires_grad_(False)
        with torch.no_grad():
            for i in range(n_iter):
                x = x / x.norm()
                x = self.apply_adjoint(self.apply_forward(x))
        return (x.norm().sqrt().item())

class gaussian_downsample(nn.Module):
    '''
    downsampling module with Gaussian filtering
    ''' 
    def __init__(self, kernel_size, sigma, stride, device):
        super().__init__()
        self.gauss = nn.Conv2d(1, 1, kernel_size, stride=stride, groups=1, 
                                bias=False)     
        self.gauss_transposed = nn.ConvTranspose2d(1, 1, kernel_size, 
                                stride=stride, groups=1, bias=False)     
        gaussian_weights = self.init_weights(kernel_size, sigma)
        self.gauss.weight.data = gaussian_weights.to(device)
        self.gauss.weight.requires_grad_(False)
        self.gauss_transposed.weight.data = gaussian_weights.to(device)
        self.gauss_transposed.weight.requires_grad_(False)
        
    def forward(self, x):
        return self.gauss(x)
    
    def transposed(self,x):
        return self.gauss_transposed(x)

    def init_weights(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance))*torch.exp(
                    -torch.sum((xy_grid - mean)**2., dim=-1)/(2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)


