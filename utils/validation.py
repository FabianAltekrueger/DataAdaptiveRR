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
from utils.validate_coarse_to_fine import ValidateCoarseToFine

def validate(reco, data, dir_name, save_name, task, **kwargs):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_hyperparameters = 2
    
    if task == 'SiC':
        p1_init = 0.05
        p2_init = 1
    elif task == 'MRI':
        p1_init = 0.01
        p2_init = 5
    elif task == 'CT':
        p1_init = 40000
        p2_init = 5
    
    p2_max = 50
    gamma_stop = 1.1

    # reconstruction map wrapper for batch
    reconstruction_map_wrapper = ReconstructionMap(reconstruction_map=reco, 
                            data=data, task=task, n_hyperparameters=n_hyperparameters, 
                            device=DEVICE)

    freeze_p2 = (reconstruction_map_wrapper.n_hyperparameters == 1)
    validator = ValidateCoarseToFine(reconstruction_map_wrapper.eval, 
                        dir_name=dir_name, exp_name=save_name, 
                        p1_init=p1_init, p2_init=p2_init, freeze_p2=freeze_p2,
                        gamma_stop=gamma_stop,p2_max=p2_max, **kwargs)
    validator.run()
                 
class ReconstructionMap():
    def __init__(self, reconstruction_map, data, task, n_hyperparameters=2, 
                    device='cuda:0', **kwargs):
        self.sample_reconstruction_map = reconstruction_map
        self.data_loader = data
        self.device = device
        self.task = task
        self.n_hyperparameters = n_hyperparameters
        if n_hyperparameters > 2 or n_hyperparameters < 1:
            raise ValueError('n_hyperparameters must be 1 or 2')
        
    def eval(self, p1, p2=None):
        psnr_val = torch.zeros(len(self.data_loader))
        ssim_val = torch.zeros(len(self.data_loader))
        n_iter_val = torch.zeros(len(self.data_loader))
        for idx, batch in tqdm(enumerate(self.data_loader)):
            if self.task in ['SiC','CT']:
                gt = batch[1].to(self.device)
                obs = batch[0].to(self.device)
                #reconstruction
                if self.n_hyperparameters == 1:
                    x, psnr_, ssim_, n_iter_, _ = self.sample_reconstruction_map(obs, 
                                                p1, p2=0, x_gt=gt)
                else:
                    x, psnr_, ssim_, n_iter_, _ = self.sample_reconstruction_map(obs, 
                                                p1, p2, x_gt=gt)
            else:
                gt = batch['x'].to(self.device)
                obs = batch['y'].to(self.device)
                mask = batch['mask'].to(self.device)
                smap = batch['smaps'].to(self.device)
                #reconstruction
                if self.n_hyperparameters == 1:
                    x, psnr_, ssim_, n_iter_, _ = self.sample_reconstruction_map(obs, 
                                                mask, smap, p1, p2=0, x_gt=gt)
                else:
                    x, psnr_, ssim_, n_iter_, _ = self.sample_reconstruction_map(obs, 
                                                mask, smap, p1, p2, x_gt=gt)
            psnr_val[idx] = psnr_
            n_iter_val[idx] = n_iter_
            ssim_val[idx] = ssim_
        return(psnr_val.mean().item(), ssim_val.mean().item(), n_iter_val.mean().item())

        
