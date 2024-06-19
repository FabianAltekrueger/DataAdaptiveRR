# This code belongs to the paper
#
# Sebastian Neumayer and Fabian Altekrüger (2024)
# Stability of Data-Dependent Ridge-Regularization for Inverse Problems
#
# Please cite the paper, if you use the code.
# The file is a modified version from 
# 
# A. Goujon, S. Neumayer, P. Bohra, S. Ducotterd, and M. Unser
# A Neural-Network-based Convex Regularizer for Inverse Problems
# IEEE Transactions on Computational Imaging, 9:781–795, 2023
# (https://github.com/axgoujon/convex_ridge_regularizer)

import torch
import torch.nn as nn

class mri_operators(nn.Module):
    '''
    Class which creates the forward operators as torch module
    Choose between single-coil and multi-coil setup
    '''
    def __init__(self, setting):
        super().__init__()
        self.setting = setting
    
    def apply_forward(self, x, mask, smap=None):
        if self.setting == 'single':
            y = torch.fft.fft2(x, norm='ortho')*mask.long()
        else:
            y = torch.fft.fft2(x*smap, norm='ortho')*mask.long()
        return y
    
    def apply_adjoint(self, x, mask, smap=None):
        if self.setting == 'single':
            y = torch.real(torch.fft.ifft2(x*mask.long(), norm='ortho'))
        else:
            smap_conj = torch.conj(smap)
            y = torch.sum(torch.real(torch.fft.ifft2(x*mask.long(), norm='ortho')
                                    *smap_conj), dim=1, keepdim=True)
        return y
        
    def get_op_norm(self, mask, smap=None, n_iter=15):
        x = torch.rand((1, 1, mask.shape[2],mask.shape[3]),device=mask.device)
        H = lambda x: self.apply_forward(x,mask,smap)
        Ht = lambda x: self.apply_adjoint(x,mask,smap)
        with torch.no_grad():
            for i in range(n_iter):
                x = x / x.norm()
                x = Ht(H(x))
        return (x.norm().sqrt().item())
