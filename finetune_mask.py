# This code belongs to the paper
#
# Sebastian Neumayer and Fabian Altekrüger (2024)
# Stability of Data-Dependent Ridge-Regularization for Inverse Problems
#
# Please cite the paper, if you use the code.

import argparse
import json
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from utils import dataset
from dival import get_standard_dataset
from models import utils_model as utils
from models import deep_equilibrium
from models.swinir import SwinIR

class Trainer:
    def __init__(self, config, config_net, device):
        self.config = config
        self.config_net = config_net
        self.device = device
        self.noise_test = config_net['noise_test']
        self.noise_range = config_net['noise_range']
        self.perturb_range = config_net['perturb_range']

        self.epochs = config['training_options']['epochs']

        # Datasets and Dataloaders
        self.batch_size = config['training_options']['batch_size']
        if args.finetune_mri:
            # set the data path to the folder of your MRI data
            data_path = f'utils/images/fastMRI/singlecoil_acc4/pd/val_images'
            self.train_dataset = dataset.fastMRI(coil='single', data_dir = data_path)
            self.train_dataloader = DataLoader(self.train_dataset,
                    batch_size=self.batch_size, shuffle=True, 
                    num_workers=1, drop_last=True)
        else:
            lodopab = get_standard_dataset('lodopab', impl='astra_cuda')
            train = lodopab.create_torch_dataset(part='train',
                            reshape=((1,) + lodopab.space[0].shape,
                            (1,) + lodopab.space[1].shape))
            self.train_dataset = [train[3][1],train[5][1],train[8][1],
                        train[11][1],train[37][1],train[75][1]]
            self.train_dataloader = DataLoader(self.train_dataset, 
                    batch_size=self.batch_size, shuffle=True, 
                    num_workers=1, drop_last=True)
        
        self.val_dataset = dataset.BSD500(config['training_options']['val_data_file'])        
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, 
                    shuffle=False, num_workers=1)
        self.validate_max = 0
        
        # build the pretrained ridge-regularizer
        self.model = utils.load_model(config_net['model'], epoch=6000, device=self.device)
        self.model.update_integrated_params()
        # updating the lipschitz constant
        self.model.conv_layer.spectral_norm(mode='power_method', n_steps=100)
        
        # fixed model for initialization of mask input
        self.model_fix = utils.load_model(config_net['model'], epoch=6000, device=self.device)
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
                config_net['training_options']['fixed_point_solver_fw_params'], 
                config_net['training_options']['fixed_point_solver_bw_params'])
        self.denoise_fix = deep_equilibrium.DEQFixedPoint(self.model_fix, 
                {'max_iter': 250, 'tol': 1e-4}, {'max_iter': 100, 'tol': 1e-3})

        # Loss
        self.criterion = torch.nn.L1Loss(reduction='sum')

        # checkpoints
        self.save_name = config['save_name']
        self.path = config['cur_net']
        config_save_path = os.path.join(self.path, f'config_{self.save_name}.json')
        with open(config_save_path, 'w') as handle:
            json.dump(getattr(self, f'config'), handle, indent=4, sort_keys=True)

    def set_optimization(self):
        optimizer = torch.optim.Adam
        params_dicts = []
        lr = self.config['optimization']['lr']
        params_dicts.append({'params': self.mask_NN.parameters(), 'lr': lr['mask_net']})
        self.optimizer = optimizer(params_dicts)
            
    def train(self):
        # set the optimizer
        self.set_optimization()
        
        step_val = 0
        tbar0 = tqdm(range(self.epochs))
        for ep in tbar0:
            tbar = tqdm(self.train_dataloader, leave=False)
            for batch_idx, img in enumerate(tbar):
                self.optimizer.zero_grad()
                
                data = img.to(self.device) if args.finetune_ct else img['x'].to(self.device)
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
                output = self.denoise(noisy_data, sigma = sigma)
                
                data_fidelity = (self.criterion(output, data))
                data_fidelity.backward()
                self.optimizer.step()
                
                tbar.set_description(f'TotalLoss {data_fidelity:.5f}')
                
            # validation
            if ep%2 == 0:
                step_val += 1
                self.validation(ep,batch_idx,step_val)
        
    def validation(self,ep,idx,step_val):
        loss_val = []
        psnr_val = []
        ssim_val = []

        tbar_val = tqdm(self.val_dataloader, leave=False)
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
        
        # save model
        if self.validate_max < np.mean(psnr_val):
            self.validate_max = np.mean(psnr_val)
            filename = f'{self.path}/{save_name}_best_state.pth'
        else:
            filename = f'{self.path}/{save_name}_last_state.pth'
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
    parser.add_argument('-fmri','--finetune_mri', action='store_true', 
                            help='Finetune the mask net for MRI task.')
    parser.add_argument('-fct','--finetune_ct', action='store_true',
                            help='Finetune the mask for CT task.')
    parser.add_argument('--rr', type=str, default='crr', 
                            choices=['crr', 'wcrr'],
                            help='Choose between type of convexity.')
    args = parser.parse_args()
    
    
    if args.finetune_mri:
        save_name = 'finetune_mri'
        config_path = 'configs/config_finetune_mri.json'
    elif args.finetune_ct:
        save_name = 'finetune_ct'
        config_path = 'configs/config_finetune_ct.json'
    else:
        raise ValueError('Please specify, for which dataset you want to finetune.')
    
    config = json.load(open(config_path))
    if args.rr == 'crr':
        config['cur_net'] = 'trained_models/SwinIR_crr' 
    else:
        config['cur_net'] = 'trained_models/SwinIR_wcrr' 
    
    net_path = config['cur_net']
    config_net = json.load(open(f'{net_path}/config.json'))
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer_inst = Trainer(config, config_net, device)

    # load weights and start finetuning
    weights = torch.load(f'{net_path}/checkpoint_best_state.pth')
    trainer_inst.mask_NN.load_state_dict(weights['state_dict_mask'])
    trainer_inst.model.load_state_dict(weights['state_dict'])
    trainer_inst.train()
    
    trainer_inst.writer.flush()
    trainer_inst.writer.close()
