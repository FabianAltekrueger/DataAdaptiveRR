# This code belongs to the paper
#
# Sebastian Neumayer and Fabian Altekrüger (2024)
# Stability of Data-Dependent Ridge-Regularization for Inverse Problems
#
# Please cite the paper, if you use the code.

import torch
import torch.nn as nn
import odl
from odl.contrib import torch as odl_torch
import dival
from dival import get_standard_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ct_operators(nn.Module):
    '''
    Class which creates the forward operators as torch module
    Choose the setting between low-dose or limited-angle CT
    '''
    def __init__(self,setting='lowdose'):
        super().__init__()
        dataset = get_standard_dataset('lodopab', impl='astra_cuda')
        if setting == 'limited':
            dataset = dival.datasets.angle_subset_dataset.AngleSubsetDataset(dataset,
	                   slice(100,900),impl='astra_cuda')   
        A = dataset.ray_trafo
        AT = A.adjoint
        
        #create FBP
        filter_type = 'Hann'
        frequency_scaling = 0.641025641025641
        fbp = odl.tomo.fbp_op(A, filter_type=filter_type, frequency_scaling=frequency_scaling)
        
        #calculate relevant operator norms:
        ATA = odl.operator.operator.OperatorComp(AT,A)
        self.op_norm_ATA = odl.power_method_opnorm(ATA, maxiter=200)
        
        # Wrap ODL operators as nn modules 
        self.A_op_layer = odl_torch.OperatorModule(A).to(device)
        self.AT_op_layer = odl_torch.OperatorModule(AT).to(device)
        self.fbp_op_layer = odl_torch.OperatorModule(fbp).to(device)
                
    def apply_forward(self, x):
        return self.A_op_layer(x)
    
    def apply_adjoint(self, y):
        return self.AT_op_layer(y)
    
    def apply_fbp(self, y):
        return self.Adag_op_layer(y)

