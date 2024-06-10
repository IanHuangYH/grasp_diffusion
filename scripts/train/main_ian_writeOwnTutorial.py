import torch
import numpy as np

import theseus as th
from theseus.geometry.so3 import SO3

from se3dif.models.grasp_dif import GraspDiffusionFields
from se3dif.models.loader import load_pointcloud_grasp_diffusion

def GetTrainData(nTrainDataNum: int) -> torch.Tensor:
    # R_z: along (0,0,1) rotate pi/4, pi*3/4, pi*5/4, pi*7/4 R^6 data
    Rotate_theta_data: torch.Tensor = torch.Tensor([[0.0, 0.0, np.pi/4],
                                                    [0.0, 0.0, np.pi*3/4],
                                                    [0.0, 0.0, np.pi*-3/4],
                                                    [0.0, 0.0, np.pi*-1/4]])
    Translate_data : torch.Tensor = torch.Tensor([  [1., 1., 0.],
                                                    [-1., 1., 0.],
                                                    [-1., -1., 0.],
                                                    [1., -1., 0.]])

    # SE(3) data
    R_data : torch.Tensor = SO3().exp_map(Rotate_theta_data).to_matrix() #DataNum * 3 * 3
    # DataNum * 4 * 4
    H_data : torch.Tensor = torch.eye(4)[None,...].repeat(nTrainDataNum, 1, 1)
    H_data[:, :3, :3] = R_data
    H_data[:, :3, -1] = Translate_data

    return Rotate_theta_data, Translate_data, H_data


def SetArgOfTrainModel() -> dict:
    args = {}
    args['device'] = torch.device('cpu')
    args['NetworkSpecs'] = {'feature_encoder': {'enc_dim': 132, 
                                                'in_dim': 3, 
                                                'out_dim': 7, 
                                                'dims': [512, 512, 512, 512, 512, 512, 512, 512], 
                                                'dropout': [0, 1, 2, 3, 4, 5, 6, 7], 
                                                'dropout_prob': 0.2, 
                                                'norm_layers': [0, 1, 2, 3, 4, 5, 6, 7], 
                                                'latent_in': [...], 
                                                'xyz_in_all': False, 
                                                'use_tanh': False, 
                                                'latent_dropout': False, 
                                                'weight_norm': True}, 
                            'encoder':          {'latent_size': 132, 'hidden_dim': 512}, 
                            'points':           {'n_points': 30, 'loc': [0.0, 0.0, 0.5], 'scale': [0.7, 0.5, 0.7]}, 
                            'decoder':          {'hidden_dim': 512}}
    return args


## 1. Dataset
nTrainDataNum = 4
RotTheta_data, TransData, HData = GetTrainData(nTrainDataNum)

## 2. set model
args_train = SetArgOfTrainModel()
model = load_pointcloud_grasp_diffusion(args_train)

## 3. train model (Forward process and learn denoise)


## 4. sample based on trained model