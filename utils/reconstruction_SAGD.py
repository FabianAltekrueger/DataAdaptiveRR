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
from tqdm import tqdm
import math

def SAGD_Recon(y, model, lmbd, sig, H=None, Ht=None, Ldata=None, 
                data_grad=None,**kwargs):
    '''
    Implementation of an averaged gradient descent with restart technique
    proposed in
    B. O’Donoghue and E. Candès. 
    Adaptive restart for accelerated gradient schemes. 
    Foundations of Computational Mathematics, 15(3):715–732, 2015
    
    Input: observation (y), ridge regularizer (model), regularzation strength (lmbd)
            noise standard deviation (mu), operators (H, Ht), 
            Lipschitz constant of the gradient of the data-fidelty term (L)
            gradient of data-fidelity term (data_grad)
    Output: reconstruction x
    '''
    
    max_iter = kwargs.get('max_iter', 1000)
    tol = kwargs.get('tol', 1e-5)
    x_init = kwargs.get('x_init', None)
    enforce_positivity = kwargs.get('enforce_positivity', True)
    
    # initial value: noisy image
    if x_init is not None:
        x = torch.clone(x_init).detach()
    else:
        x = torch.clone(Ht(y)).zero_().detach()

    z = torch.clone(x)
    x_old = torch.clone(x)
    z_old = torch.clone(x)
    t = 1
    t_old = 1

    sigma = torch.ones((1, 1, 1, 1), device=x.device) * sig

    model.mu = None
    model.scaling = None
    scaling = model.get_scaling(sigma=sigma)
    mu = model.get_mu()
    model.mu = mu
    model.scaling = scaling
    model.get_mask(x)
    
    L = Ldata + model.mu * lmbd
    
    restart_count = 0
    
    def grad_func(xx):
        r_grad = model.grad(xx, sigma)
        d_grad = data_grad.compute_grad(xx)
        return (d_grad + lmbd * r_grad)

    beta = 1.001
    
    pbar = tqdm(range(max_iter),leave=False)
    for i in pbar:
        z = x + (t_old - 1)/t * (x - x_old)
        grad_z = grad_func(z)
        crit = (torch.sum((grad_z*(z - z_old))) + 1/2 * (lmbd * beta * 
                                    torch.sum((z - z_old)**2)).item())
        if crit > 0:
            z = x.clone()
            t = 1
            t_old = 1
            restart_count += 1
            grad_z = grad_func(z)

        x_old = torch.clone(x)      
        x = z - 1/L * grad_z
        if enforce_positivity:
            x = torch.clamp(x, 0, None)
        t_old = t
        t = (1 + math.sqrt(1 + 4 * t**2))/2
        z_old = torch.clone(z)

        # relative change of norm for terminating
        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
        if res < tol:
                break
        pbar.set_description(f'res: {res:.2e}')
    return(x, i)
