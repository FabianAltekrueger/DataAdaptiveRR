# This code belongs to the paper
#
# Sebastian Neumayer and Fabian Altekrüger (2024)
# Stability of Data-Dependent Ridge-Regularization for Inverse Problems
#
# Please cite the paper, if you use the code.

import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import skimage.io as io
import matplotlib.pyplot as plt
import pandas as pd

from models import utils_model, deep_equilibrium
from models.swinir import SwinIR
from utils import dataset

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--rr', type=str, default='crr', 
                        choices=['crr', 'wcrr'],
                        help='Choose between type of convexity.')
parser.add_argument('-m', '--use_mask', action='store_false',
                        help='Choose if you want to deactivate the mask.')
parser.add_argument('-n', '--noise', type=int, default='25', 
                        help='Noise level.')
parser.add_argument('--vis', action='store_true', 
                        help='Visualize mask and responses.')
parser.add_argument('--test', action='store_true',
                        help='Test the method for BSD68.')
args = parser.parse_args()

# name of path
dir_name = 'trained_models/SwinIR_crr' if args.rr == 'crr' else 'trained_models/SwinIR_wcrr'
save_name = f'mask_{args.rr}' if args.use_mask else f'{args.rr}'

# load default
config_path = f'{dir_name}/config.json'
config = json.load(open(config_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load ridge regularizer
model = utils_model.load_model(config['model'], epoch=6000, device=device)
model.update_integrated_params()
model_fix = utils_model.load_model(config['model'], epoch=6000, device=device)
model_fix.update_integrated_params()
sn_pm = model.conv_layer.spectral_norm(mode='power_method', n_steps=100)

# create mask model with included postscaling to [0,1]
mask_nn = SwinIR(in_chans=1,out_chans=model.num_channels,
                upscale=1,window_size=8,img_range=1., depths=[6, 6, 6, 6],
                embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                upsampler='pixelshuffledirect').to(device)
mask_nn.eval()

# load weights
weights = torch.load(f'{dir_name}/checkpoint_best_state.pth')
mask_nn.load_state_dict(weights['state_dict_mask'])
model.load_state_dict(weights['state_dict'])
    
# set the models
def constant_mask(inp):
        return torch.ones_like(inp)
model_fix.set_mask_net(constant_mask)
denoise_fix = deep_equilibrium.DEQFixedPoint(model_fix, 
            {'max_iter': 250, 'tol': 1e-4}, {'max_iter': 100, 'tol': 1e-3})
denoise = deep_equilibrium.DEQFixedPoint(model, 
            config['training_options']['fixed_point_solver_fw_params'],
            config['training_options']['fixed_point_solver_bw_params'])

# load image
im_clean = torch.tensor(io.imread(f'utils/images/BSD/Set68/test003.png'),
                        device=device).unsqueeze(0).unsqueeze(0)/255
sigma = torch.tensor(args.noise,device=device).view(1,1,1,1)
noise = sigma/255 * torch.randn_like(im_clean)
im_noisy = im_clean + noise

# reconstruction
with torch.no_grad():
    target_estimate = denoise_fix(im_noisy,sigma=sigma)
    def mask_net(inp):
        return mask_nn(target_estimate)
    model.set_mask_net(mask_net)        
    output = denoise(im_noisy,sigma=sigma) if args.use_mask else target_estimate
tmp = (255 * torch.clip(output,0,1)).detach().cpu().squeeze().numpy().astype(np.uint8)
io.imsave(f'results/denoising/reconstruction_{save_name}.png', tmp)
        
# reconstruct and visualize responses
if args.vis:
    with torch.no_grad():
        mask_pred = torch.mean(mask_net(target_estimate),1,keepdim=True)         
        # compute responses
        a = torch.max(mask_pred)
        reg_gt = torch.mean(a*model.integrate_activation(model.conv_layer(im_clean),sigma),dim=1)
        reg_noisy = torch.mean(a*model.integrate_activation(model.conv_layer(im_noisy),sigma),dim=1)
        reg_init = torch.mean(a*model.integrate_activation(model.conv_layer(target_estimate),sigma),dim=1)
        reg_mask = torch.mean(mask_net(target_estimate)*model.integrate_activation(model.conv_layer(im_noisy),sigma),dim=1)
    # save images
    plt.imsave(f'results/denoising/{args.rr}_mask_mean.png', 
                    mask_pred.detach().cpu().squeeze().numpy(),cmap='gray')
    plt.imsave(f'results/denoising/response_{args.rr}_gt.png', 
                    reg_gt.detach().cpu().squeeze().numpy(),cmap='gray_r')
    plt.imsave(f'results/denoising/response_{args.rr}_noisy.png', 
                    reg_noisy.detach().cpu().squeeze().numpy(),cmap='gray_r')
    plt.imsave(f'results/denoising/response_{args.rr}_init.png', 
                    reg_init.detach().cpu().squeeze().numpy(),cmap='gray_r')
    plt.imsave(f'results/denoising/response_{args.rr}_mask.png', 
                    reg_mask.detach().cpu().squeeze().numpy(),cmap='gray_r')

# if you test the method on BSD68 dataset
if args.test:
    # load data
    test_dataset = dataset.BSD500(f'utils/images/test_BSD.h5')
    test_dataloader = DataLoader(test_dataset, batch_size=1, 
                    shuffle=False, num_workers=1)
    # set noise levels for testing
    Sigma = [5, 15, 25]
    columns = ['sigma', 'image_id', 'psnr', 'ssim']
    df = pd.DataFrame(columns=columns)
    df_mask = pd.DataFrame(columns=columns)
    sigma = torch.tensor(Sigma).to(device).view(-1,1,1,1)
    psnrs = [[],[],[]]; ssims = [[],[],[]]
    for idx, im in enumerate(tqdm(test_dataloader)):
        im = im.to(device).repeat(len(Sigma),1,1,1)
        im_noisy = im + sigma/255*torch.randn_like(im)
        reco = torch.tensor([],device=device)
        
        # reconstruct for each noise level
        for i in range(len(im_noisy)):
            tmp = im_noisy[i,...].unsqueeze(0)
            tmp_sigma = sigma[i,...].unsqueeze(0)
            with torch.no_grad():
                target_estimate = denoise_fix(tmp,sigma=tmp_sigma)
                def mask_net(inp):
                    return mask_nn(target_estimate)
                model.set_mask_net(mask_net)
                output = denoise(tmp,sigma=tmp_sigma) if args.use_mask else target_estimate
                reco = torch.cat([reco,output],dim=0)
        
        # evaluate for each noise level
        for i, sigma_n in enumerate(Sigma):
            psnr_ = psnr(im[0].unsqueeze(0).cpu(), reco[i].unsqueeze(0).cpu(), data_range=1).item()
            psnrs[i].append(psnr_)
            ssim_ = ssim(im[0].unsqueeze(0).cpu(), reco[i].unsqueeze(0).cpu(), data_range=1).item()
            ssims[i].append(psnr_)
            df = pd.concat([df, pd.DataFrame([[sigma_n, idx, psnr_, ssim_]], columns=columns)])
    for i, sigma_n in enumerate(Sigma):
        df = pd.concat([df, pd.DataFrame([[sigma_n, 99, np.mean(psnrs[i]), np.mean(ssims[i])]], columns=columns)])
        df.to_csv(f'results/denoising/test_{save_name}.csv')
    
