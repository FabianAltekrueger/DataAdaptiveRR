# This code belongs to the paper
#
# Sebastian Neumayer and Fabian Altekrüger (2024)
# Stability of Data-Dependent Ridge-Regularization for Inverse Problems
#
# Please cite the paper, if you use the code.

import argparse
import json
import pandas
import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import numpy as np
import skimage.io as io

from utils.SiC_utils import superresolution_operators, imread
from models.swinir import SwinIR
from models import utils_model
import utils.validation as validation
from utils.reconstruction_SAGD import SAGD_Recon

import warnings
warnings.filterwarnings('ignore')

def get_reconstruction_map(scale_factor,operators):
    # set up the superresolution operators
    H = operators.apply_forward
    Ht = operators.apply_adjoint
    op_norm = operators.get_op_norm([600,600],100) # = 1/scale_factor
    
    # define the gradient of the data fidelity term
    # D(Hx,y) = 1/2 ||Hx - y||**2
    class data_fidelity_grad():
        def __init__(self, H, Ht, obs):
            self.H = H
            self.Ht = Ht
            self.obs = obs
        def compute_grad(self,xx):
            data_grad = self.Ht(self.H(xx) - self.obs)
            return data_grad
    
    # without mask
    if not args.use_mask:
        model_name = 'CRR-CNN' if args.rr == 'crr' else 'WCRR-CNN'
        model = utils_model.load_model(f'trained_models/{model_name}',  
                            epoch=6000, device=DEVICE)
        model.eval()
        sn_pm = model.conv_layer.spectral_norm(mode='power_method', n_steps=1000)
        
        def reconstruction_map(y, lam, sig, x_init=None, x_gt=None):
            data_f_grad = data_fidelity_grad(H=H,Ht=Ht,obs=y)
            Ldata = op_norm**2
            x, n_iter = SAGD_Recon(y, model, lmbd=lam, sig=sig, 
                        H=H, Ht=Ht, x_init=x_init, Ldata=Ldata, 
                        data_grad=data_f_grad)
            if x_gt is not None:
                psnr_ = psnr(x[...,40:-40,40:-40], x_gt[...,40:-40,40:-40], data_range=1)
                ssim_ = ssim(x[...,40:-40,40:-40], x_gt[...,40:-40,40:-40], data_range=1)
            else:
                psnr_ = None; ssim_ = None
            return(x, psnr_, ssim_, n_iter, None)
                
    # with mask
    else:
        path = 'trained_models/SwinIR_crr' if args.rr == 'crr' else 'trained_models/SwinIR_wcrr'
        config_path_net = f'{path}/config.json'
        config_net = json.load(open(config_path_net))
        
        # load models
        model = utils_model.load_model(config_net['model'], epoch=6000, device=DEVICE)
        model.update_integrated_params()
        model_fix = utils_model.load_model(config_net['model'], epoch=6000, device=DEVICE)
        model_fix.update_integrated_params()
        sn_pm = model.conv_layer.spectral_norm(mode='power_method', n_steps=100)
        
        # set mask for preprocessing model
        def constant_mask(inp):
            return torch.ones(inp.shape,device = inp.device)
        model_fix.set_mask_net(constant_mask)
        
        # define mask network with included postscaling to [0,1]
        mask_NN = SwinIR(in_chans=1,out_chans=model.num_channels,
                upscale=1,window_size=8,img_range=1., depths=[6, 6, 6, 6],
                embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                upsampler='pixelshuffledirect').to(DEVICE)
        
        # load weights
        weights = torch.load(f'{path}/checkpoint_best_state.pth')
        mask_NN.load_state_dict(weights['state_dict_mask'])
        mask_NN.eval()
        model.load_state_dict(weights['state_dict'])
        
        # load optimal sigma for preprocessing step
        hyperparams_csv = pandas.read_csv(f'{dir_name}/validation_scores_{args.rr}.csv')
        sig_pre = hyperparams_csv.loc[hyperparams_csv['psnr'].idxmax()]['p2']
        
        def reconstruction_map(y, lam, sig, x_init=None, x_gt=None):
            data_f_grad = data_fidelity_grad(H=H,Ht=Ht,obs=y)
            # preprocessing for initial reconstruction
            Ldata = op_norm**2
            x, n_iter1 = SAGD_Recon(y, model_fix, lmbd=lam, sig=sig_pre, 
                        H=H, Ht=Ht, x_init=x_init, Ldata=Ldata, 
                        data_grad=data_f_grad)

            # define mask for intial reconstruction
            def mask_net(inp):
                return mask_NN(x)
            model.set_mask_net(mask_net)
            x_est = x.clone()
            mask_pred = torch.mean(mask_net(x_est),1,keepdim=True)

            # reconstruction with data-adaptive regularizer
            x, n_iter2 = SAGD_Recon(y, model, lmbd=lam, sig=sig, H=H, 
                                    Ht=Ht, Ldata=Ldata, x_init = x_est, 
                                    data_grad = data_f_grad)
            if x_gt is not None:
                psnr_ = psnr(x[...,40:-40,40:-40], x_gt[...,40:-40,40:-40], data_range=1)
                ssim_ = ssim(x[...,40:-40,40:-40], x_gt[...,40:-40,40:-40], data_range=1)
            else:
                psnr_ = None; ssim_ = None
            return(x, psnr_, ssim_, n_iter1+n_iter2, mask_pred)
            
    return(reconstruction_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rr', type=str, default='crr', 
                            choices=['crr', 'wcrr'],
                            help='Choose between type of convexity.')
    parser.add_argument('-m', '--use_mask', action='store_false',
                            help='Choose if you want to deactivate the mask.')
    parser.add_argument('-s', '--scale_factor', type=int, default=4,
                            help='Choose the superresolution factor.')
    parser.add_argument('-val','--validate', action='store_true',
                            help='Choose if you want to start a grid search \
                                  on a validation set for determining the \
                                  regularization scale and noise level.')
    args = parser.parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    # define operators
    operators = superresolution_operators(args.scale_factor,device=DEVICE)
    H = operators.apply_forward
    Ht = operators.apply_adjoint
    
    # define saving path and reconstruction map
    dir_name = 'results/superresolution'
    save_name = f'mask_{args.rr}' if args.use_mask else f'{args.rr}'
    reco_map = get_reconstruction_map(scale_factor=args.scale_factor,
                                    operators = operators)
    
    if not args.validate:
        # load hyperparameters for reconstruction
        if args.use_mask:
            params_path = f'{dir_name}/validation_scores_mask_{args.rr}.csv'
        else:
            params_path = f'{dir_name}/validation_scores_{args.rr}.csv'
        hyperparams_csv = pandas.read_csv(params_path)
        lam = hyperparams_csv.loc[hyperparams_csv['psnr'].idxmax()]['p1']
        sigma = hyperparams_csv.loc[hyperparams_csv['psnr'].idxmax()]['p2']
        
        # load images
        gt = imread(f'utils/images/SiC/img_test_material.png',device=DEVICE)
        obs = H(gt)
        obs = obs + 0.01*torch.randn_like(obs)
        
        # reconstruction
        with torch.no_grad():
            x,psnr_,ssim_,iter,mask_pred = reco_map(obs,lam,sigma,x_gt=gt)
            
        # save results
        tmp = torch.clip(x,0,1).squeeze().cpu().numpy()
        tmp = (tmp*255).astype(np.uint8)
        io.imsave(f'{dir_name}/reco_{save_name}.png',tmp)
        if args.use_mask:
            tmp = mask_pred.squeeze().detach().cpu().numpy()
            tmp = (tmp*255).astype(np.uint8)
            io.imsave(f'{dir_name}/{save_name}.png',tmp)
            
    else:
        # validation for determining the regularization strength and the
        # noise level
        hr = imread('utils/images/SiC/img_learn_material.png',device=DEVICE)
        lr = H(hr)
        lr = lr + 0.01*torch.randn_like(lr)
        data = [[lr,hr]]
        with torch.no_grad():
            validation.validate(reco=reco_map, data=data, dir_name = dir_name,
                                save_name = save_name, task='SiC')
