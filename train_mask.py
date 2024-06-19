# This code belongs to the paper
#
# Sebastian Neumayer and Fabian Altekrüger (2024)
# Stability of Data-Dependent Ridge-Regularization for Inverse Problems
#
# Please cite the paper, if you use the code.

import json
import os
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from utils import dataset
from models import utils_model as utils
from models import deep_equilibrium
from models.swinir import SwinIR

class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.noise_test = config['noise_test']
        self.noise_range = config['noise_range']
        self.perturb_range = config['perturb_range']

        self.epochs = config['training_options']['epochs']

        # Datasets
        self.train_dataset = dataset.BSD500(config['training_options']['train_data_file'])
        self.val_dataset = dataset.BSD500(config['training_options']['val_data_file'])
        
        # Dataloaders
        self.batch_size = config['training_options']['batch_size']
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
                    shuffle=True, num_workers=1, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, 
                    shuffle=False, num_workers=1)
        self.validate_max = 0
        
        # build the pretrained ridge-regularizer
        self.model = utils.load_model(config['model'], epoch=6000, device=self.device)
        self.model.update_integrated_params()
        # updating the lipschitz constant
        self.model.conv_layer.spectral_norm(mode='power_method', n_steps=100)
        
        # fixed model for initialization of mask input
        self.model_fix = utils.load_model(config['model'], epoch=6000, device=self.device)
        self.model_fix.update_integrated_params()
        self.model_fix.eval()
        
        def constant_mask(inp):
            return torch.ones_like(inp)
        self.model_fix.set_mask_net(constant_mask)
        
        # define regularization mask network with included postscaling to [0,1]
        self.mask_NN = SwinIR(in_chans=1,out_chans=self.model.num_channels,
                upscale=1,window_size=8,img_range=1., depths=[6, 6, 6, 6],
                embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                upsampler='pixelshuffledirect').to(device)

        # Set the DEQ solver
        self.denoise = deep_equilibrium.DEQFixedPoint(self.model, 
                config['training_options']['fixed_point_solver_fw_params'], 
                config['training_options']['fixed_point_solver_bw_params'])
        self.denoise_fix = deep_equilibrium.DEQFixedPoint(self.model_fix, 
                {'max_iter': 250, 'tol': 1e-4}, {'max_iter': 100, 'tol': 1e-3})

        # Loss
        self.criterion = torch.nn.L1Loss(reduction='sum')

        # checkpoints
        self.path = config['saving_path']
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        config_save_path = os.path.join(self.path, f'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(getattr(self, f'config'), handle, indent=4, sort_keys=True)
        
        # tensorboard
        writer_dir = os.path.join(self.path, 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)

    def set_optimization(self, pretrained):
        optimizer = torch.optim.Adam
        params_dicts = []
        lr = self.config['optimization']['lr']
        if pretrained:
            #then adjust the ridge regularizer
            params_dicts.append({'params': [self.model.mu_], 'lr': lr['mu']})
            params_dicts.append({'params': [self.model.spline_scaling.coefficients], 'lr': lr['spline_scaling']})
        else:
            #then keep the ridge regularizer fixed and just adjust the noise level
            params_dicts.append({'params': [self.sig], 'lr': lr['sig']})

        params_dicts.append({'params': self.mask_NN.parameters(), 'lr': lr['mask_net']})
        self.optimizer = optimizer(params_dicts)
        
        # scheduler
        if self.config['training_options']['scheduler']['use']:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
                    start_factor=1,
                    end_factor=self.config['training_options']['scheduler']['end_factor'],
                    total_iters=self.config['training_options']['scheduler']['n_times'], 
                    verbose=False)
            
    def train(self,pretrained):
        # set the optimizer for the correct training step
        self.set_optimization(pretrained)
        
        step = 0; step_val = 0
        tbar0 = tqdm(range(self.epochs))
        for ep in tbar0:
            tbar = tqdm(self.train_dataloader, leave=False)
            for img in tbar:
                step += 1
                self.optimizer.zero_grad()
                
                data = img.to(self.device)
                # draw random noise level and random perturbation level
                sigma = (self.noise_range[1]-self.noise_range[0])*torch.rand(
                            (data.shape[0], 1, 1, 1), device=data.device) + self.noise_range[0]
                perturb = (self.perturb_range[1]-self.perturb_range[0])*torch.rand(
                            (data.shape[0], 1, 1, 1), device=data.device) + self.perturb_range[0]
                # create noisy data
                noise = sigma/255 * torch.randn_like(data)
                noisy_data = data + noise
                
                # estimate fixed point for mask estimate
                def constant_mask(inp):
                    return torch.ones_like(inp)
                with torch.no_grad():
                    self.model_fix.set_mask_net(constant_mask)
                    target_estimate = self.denoise_fix(noisy_data, 
                                    sigma = torch.abs(sigma+perturb))
                # set mask network
                def mask_net(inp):
                    return self.mask_NN(target_estimate)
                self.model.set_mask_net(mask_net)
                
                if not pretrained:
                    output = self.denoise(noisy_data,
                            sigma=self.sig*torch.ones((data.shape[0], 1, 1, 1),device=device))
                else:
                    output = self.denoise(noisy_data, sigma = sigma)
                
                data_fidelity = (self.criterion(output, data))
                data_fidelity.backward()
                self.optimizer.step()
                
                tbar.set_description(f'TotalLoss {data_fidelity:.5f}')
                
                # tensorboard values
                self.writer.add_scalar(f'Loss/training_loss', data_fidelity, step)
                self.writer.add_scalars(f'Iter/iterfw', 
                        {'fw_niter_mean': self.denoise.forward_niter_mean,
                        'fw_niter_max': self.denoise.forward_niter_max}, step)
                
                # validation
                if (step)%((len(self.train_dataset)//self.batch_size)//4) == 0:
                    step_val += 1
                    self.validation(ep,step,step_val)
            
            #scheduler
            if self.config['training_options']['scheduler']['use']:
                self.scheduler.step()
        
    def validation(self,ep,idx,step_val):
        loss_val = []
        psnr_val = []
        ssim_val = []

        tbar_val = self.val_dataloader
        with torch.no_grad():
            for batch_idx, data in enumerate(tbar_val):       
                data = data.to(device)

                # simulate noisy data
                sigma = 25. * torch.ones((data.shape[0], 1, 1, 1),device=device)
                noise = 25. / 255 * torch.randn_like(data)
                noisy_data = data + noise

                # estimate fixed point for mask estimate
                def constant_mask(inp):
                    return torch.ones_like(inp)
                with torch.no_grad():
                    self.model_fix.set_mask_net(constant_mask)
                    target_estimate = self.denoise_fix(noisy_data, 
                                        sigma = sigma)
                def mask_net(inp):
                    return self.mask_NN(target_estimate)
                self.model.set_mask_net(mask_net)    
                
                # reconstruct
                output = self.denoise(noisy_data , sigma = sigma)
                loss = self.criterion(output, data)
                out_val = torch.clamp(output, 0., 1.)
                
                loss_val.append(loss.cpu().item())
                psnr_val.append(psnr(out_val, data, 1.).item())
                ssim_val.append(ssim(out_val, data, 1.).item())

        self.writer.add_scalar(f'Loss/validation_loss', np.mean(loss_val), step_val)
        self.writer.add_scalar(f'Metrics/validation_PSNR', np.mean(psnr_val), step_val)
        self.writer.add_scalar(f'Metrics/validation_SSIM', np.mean(ssim_val), step_val)
        
        # save model
        if self.validate_max < np.mean(psnr_val):
            self.validate_max = np.mean(psnr_val)
            filename = f'{self.path}/checkpoint_best_state.pth'
        else:
            filename = f'{self.path}/checkpoint_last_state.pth'
        state = {
                'epoch': ep,
                'batch': idx,
                'state_dict': self.model.state_dict(),
                'state_dict_mask': self.mask_NN.state_dict(),
                'PSNR_val': np.mean(psnr_val)
        }            
        torch.save(state, filename)
        return 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rr', type=str, default='crr', 
                            choices=['crr', 'wcrr'],
                            help='Choose between type of convexity.')
    args = parser.parse_args()
    config_path = 'configs/config_train_mask.json'
    config = json.load(open(config_path))
    if args.rr == 'crr':
        config['model'] = 'trained_models/CRR-CNN' 
        config['rho_wcvx'] = 0
    else:
        config['model'] = 'trained_models/WCRR-CNN' 
        config['rho_wcvx'] = 1
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer_inst = Trainer(config, device)

    # Start pretraining
    trainer_inst.epochs = config['pretrain_epochs']
    trainer_inst.sig = torch.tensor(15.,device=device, requires_grad = True)
    trainer_inst.noise_range = [10,10]
    trainer_inst.train(pretrained=False)
    
    #Start posttraining
    trainer_inst.noise_range = config['noise_range']
    trainer_inst.epochs = config['training_options']['epochs']
    trainer_inst.train(pretrained=True)
    
    trainer_inst.writer.flush()
    trainer_inst.writer.close()
