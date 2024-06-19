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

import sys
import os
import argparse
import numpy as np
import h5py
from fastmri.data import subsample
from fastmri.data.transforms import center_crop
import torch
import skimage.io as io
from tqdm import tqdm

# set bart path 
version = 'bart-0.8.00'
cur_wd = os.getcwd()
b_path = f'{cur_wd}/utils/{version}/python'
sys.path.insert(0, b_path)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOOLBOX_PATH']    = f'{cur_wd}/utils/{version}'
from bart import bart
        
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--single', action='store_true',
                            help='Choose if you want to generate the \
                            single-coil data.')
parser.add_argument('-m', '--multi', action='store_true',
                            help='Choose if you want to generate the \
                            multi-coil data.')
args = parser.parse_args()

# path where the fastMRI data is
loaddata_path = 'utils/images/fastMRI/knee_multicoil_val'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 0.002

if args.single:
    center_fraction_s = 0.08
    acceleration_s = 4
    mask_func_s = subsample.RandomMaskFunc(center_fractions=[center_fraction_s], accelerations=[acceleration_s])
    savedata_path_sc = f'images/fastMRI/singlecoil_acc{acceleration_s}'
    os.makedirs(savedata_path_sc, exist_ok=True)

elif args.multi:
    center_fraction_m = 0.04
    acceleration_m = 8
    mask_func_m = subsample.RandomMaskFunc(center_fractions=[center_fraction_m], accelerations=[acceleration_m])
    savedata_path_mc = f'images/fastMRI/multicoil_acc{acceleration_m}'
    os.makedirs(savedata_path_mc, exist_ok=True)


files = sorted(os.listdir(loaddata_path))
# Remove noisy data from the dataset
files.remove('file1000007.h5')
files.remove('file1000903.h5')
total_num_files = len(files)

num_val_files_each = 10
num_test_files_each = 50

print('Generating datasets')
count_pd_val = 0
count_pd_test = 0
count_pdfs_val = 0
count_pdfs_test = 0

for file_idx in tqdm(range(total_num_files)):
    loadfilename = os.path.join(loaddata_path, files[file_idx])
    filecontent = h5py.File(loadfilename, 'r')
    
    # check for fat supression (PDFS) or not (PD)
    if (dict(filecontent.attrs)['acquisition'] == 'CORPD_FBK'):
        data_tag = 'pd'
        if (count_pd_val < num_val_files_each):
            save_tag = 'val_images'
            count_pd_val += 1
        elif (count_pd_test < num_test_files_each):
            save_tag = 'test_images'
            count_pd_test += 1
        else:
            continue

    elif (dict(filecontent.attrs)['acquisition'] == 'CORPDFS_FBK'):
        data_tag = 'pdfs'
        if (count_pdfs_val < num_val_files_each):
            save_tag = 'val_images'
            count_pdfs_val += 1
        elif (count_pdfs_test < num_test_files_each):
            save_tag = 'test_images'
            count_pdfs_test += 1
        else:
            continue
    
    kspace = filecontent['kspace']
    slice_idx = kspace.shape[0]//2
    kspace = np.expand_dims(kspace[slice_idx,...], axis=0)
    kspace_bart = np.transpose(kspace, (0, 2, 3, 1))
    kspace = torch.tensor(kspace,device=device)
    
    # Use fully-sampled kspace data to create ground-truth images
    kspace_shifted = torch.fft.ifftshift(kspace, dim=(2,3))
    ifft_kspace_shifted = torch.fft.ifft2(kspace_shifted, dim=(2,3), norm='ortho')
    ifft_kspace = torch.fft.fftshift(ifft_kspace_shifted, dim=(2,3))
    x = torch.sqrt(torch.sum(torch.abs(ifft_kspace)**2, dim=1, keepdim=True))

    # Normalize x
    x = x/torch.max(x)

    # Cropped version of
    x_crop = center_crop(x, [320,320])

    if args.single:
        # Create Mask
        h = kspace.shape[2]
        w = kspace.shape[3]
        mask, num_low_frequencies = mask_func_s([h, w, 1])
        mask = mask[:,:,0]
        M = mask.to(device)
        M = M.expand(h,-1)
        M = M.unsqueeze(0).unsqueeze(0)
        M = torch.fft.ifftshift(M, dim=(2,3))

        # Generate singlecoil measurements
        y0_sc = torch.fft.fft2(x, dim=(2,3), norm='ortho')*M
        y0_sc += (noise_std*torch.randn_like(y0_sc) + 
                        1j*noise_std*torch.randn_like(y0_sc))
        y0_sc *= M  # mask noise
        
        # save files
        currfile_savepath_sc = f'{savedata_path_sc}/{data_tag}/{save_tag}/{files[file_idx][0:11]}'
        os.makedirs(currfile_savepath_sc, exist_ok=True)
        torch.save(x, f'{currfile_savepath_sc}/x.pt')
        tmp = (255*x.cpu().squeeze().numpy()).astype(np.uint8)
        io.imsave(f'{currfile_savepath_sc}/x.png',tmp)
        torch.save(x_crop, f'{currfile_savepath_sc}/x_crop.pt')
        torch.save(y0_sc, f'{currfile_savepath_sc}/y.pt')
        torch.save(M, f'{currfile_savepath_sc}/mask.pt')


    if args.multi:
        # Create Mask
        h = kspace.shape[2]
        w = kspace.shape[3]
        mask, num_low_frequencies = mask_func_m([h, w, 1])
        mask = mask[:,:,0]
        M = mask.to(device)
        M = M.expand(h,-1)
        M = M.unsqueeze(0).unsqueeze(0)
        M = torch.fft.ifftshift(M, dim=(2,3))
        
        # get sensitivity maps
        smaps = bart(1, 'ecalib -m1 -W -c0', kspace_bart)
        smaps = np.transpose(smaps, (0,3,1,2))
        smaps = torch.tensor(smaps)
        
        # Generate multicoil measurements
        y0_mc = torch.fft.fft2(x*smaps, dim=(2,3), norm='ortho')*M
        y0_mc += (noise_std*torch.randn_like(y0_mc) + 
                        1j*noise_std*torch.randn_like(y0_mc))
        y0_mc *= M  # mask noise
        
        # save files
        currfile_savepath_mc = f'{savedata_path_mc}/{data_tag}/{save_tag}/{files[file_idx][0:11]}'
        os.makedirs(currfile_savepath_mc, exist_ok=True)
        torch.save(x, f'{currfile_savepath_mc}/x.pt')
        io.imsave(f'{currfile_savepath_mc}/x.png',x.cpu().squeeze().numpy())
        torch.save(x_crop, f'{currfile_savepath_mc}/x_crop.pt')
        torch.save(y0_mc, f'{currfile_savepath_mc}/y.pt')
        torch.save(M, f'{currfile_savepath_mc}/mask.pt')
        torch.save(smaps, f'{currfile_savepath_mc}/smaps.pt')
