import torch
import numpy as np

import theseus as th
from theseus.geometry.so3 import SO3

from se3dif.models.grasp_dif import GraspDiffusionFields
from se3dif.models.loader import load_pointcloud_grasp_diffusion

from torch.utils.data import DataLoader, Dataset
import configargparse
from se3dif.datasets import AcronymGraspsDirectory
import scipy.spatial.transform
from se3dif.utils import to_torch, SO3_R3
from mesh_to_sdf.scan import ScanPointcloud

import matplotlib.pyplot as plt

OPT_PROJ_SE3_DENOISE_LOSS = 1
RAW_DATA_NUM = 4
EXPAND_SCALE = 100
BATCH_SIZE = 1

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

class GraspDataset(Dataset):
    def __init__(self, GraspHData: torch.Tensor):
        self.m_CleanH = GraspHData

    def __getitem__(self, index):
        return self.m_CleanH[index]
    
    def __len__(self):
        return len(self.m_CleanH)

def parse_Sample_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=str, default='12')
    p.add_argument('--n_grasps', type=str, default='10')
    p.add_argument('--obj_class', type=str, default='Mug')

    opt = p.parse_args()
    return opt

def sample_pointcloud(obj_id=0, obj_class='Mug'):
    acronym_grasps = AcronymGraspsDirectory(data_type=obj_class)
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh()

    centroid = mesh.centroid
    H = np.eye(4)
    H[:3,-1] = -centroid
    mesh.apply_transform(H)

    scan_pointcloud = ScanPointcloud()
    P = scan_pointcloud.get_hq_scan_view(mesh)


    P *= 8.
    P_mean = np.mean(P, 0)
    P = P - P_mean

    rot = scipy.spatial.transform.Rotation.random().as_matrix()
    P = np.einsum('mn,bn->bm', rot, P)

    mesh.apply_scale(8.)
    H = np.eye(4)
    H[:3,-1] = -P_mean
    mesh.apply_transform(H)
    H = np.eye(4)
    H[:3,:3] = rot
    mesh.apply_transform(H)

    return P, mesh

def PutShapeCodeMaskIntoModel(device, model):
    args_sample = parse_Sample_args()
    obj_id = int(args_sample.obj_id)
    obj_class = args_sample.obj_class
    P, mesh = sample_pointcloud(obj_id, obj_class)
    context = to_torch(P[None,...], device)
    model.set_latent(context, batch=EXPAND_SCALE * BATCH_SIZE)
    return context

class SimpleGraspTrainer():
    def __init__(self, device) -> None:
        self.m_device = device

    def __SetOptimizer(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam([
        {
            "params": model.vision_encoder.parameters(),
            "lr": 0.0005,
        },
        {
            "params": model.feature_encoder.parameters(),
            "lr": 0.0005,
        },
        {
            "params": model.decoder.parameters(),
            "lr": 0.0005,
        },
        ])
        return optimizer

    def marginal_prob_std(self, t, sigma=0.5):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))
    
    def __PerturbHData(self, Data:torch.tensor, ExpandNum: int, eps=1e-5) -> torch.tensor:
        '''Data: H with 4x4 tensor'''

        ## 1. expand H ##
        HSingle = SO3_R3(R=Data[...,:3, :3], t=Data[...,:3, -1])
        Matrix_expand_H = HSingle.to_matrix().repeat(ExpandNum,1,1)
        H_expand = SO3_R3(R=Matrix_expand_H[...,:3, :3], t=Matrix_expand_H[...,:3, -1])
        # H_expand = SO3_R3(R=Data[...,:3, :3], t=Data[...,:3, -1])
        
        ## 2. H to vector ##
        VecCleanH = H_expand.log_map()
        
        ## 3. generate noise
        random_t_step = torch.rand_like(VecCleanH[...,0], device=self.m_device) * (1. - eps) + eps
        VecNoise = torch.randn_like(VecCleanH)
        std = self.marginal_prob_std(random_t_step)
        VecPerturbed_H = VecCleanH + VecNoise * std[..., None]
        VecPerturbed_H = VecPerturbed_H.detach()
        VecPerturbed_H.requires_grad_(True)
        Matrix_PerturbData = SO3_R3().exp_map(VecPerturbed_H).to_matrix()

        ## 4. normalize generate noise
        NormalNoise = VecNoise/std[...,None]


        return Matrix_PerturbData.clone(), NormalNoise.clone(), random_t_step.clone(), VecPerturbed_H.clone()
    
    def __CalculateLoss(self, option:int, model, 
                      random_t_step: torch.tensor, NormalNoise:torch.tensor, 
                      Matrix_PerturbData:torch.tensor, VecPerturbed_H:torch.tensor):

        with torch.set_grad_enabled(True):
            energy = model(Matrix_PerturbData, random_t_step)
            grad_energy = torch.autograd.grad(energy.sum(), VecPerturbed_H,
                                            only_inputs=True, retain_graph=True, create_graph=True)[0]
        loss_fn = torch.nn.L1Loss()
        Loss = loss_fn(grad_energy, NormalNoise)/10.
            
        return Loss

    def train(self, model, TrainDataLoader: DataLoader):
        optimizer = [self.__SetOptimizer()]
        Train_Iter = 1000
        loss_trj = torch.zeros(0)

        
        for Iter in range(Train_Iter):
            for idx_Data, CleanHData in enumerate(TrainDataLoader):
                train_loss = 0.

                # Enable anomaly detection
                with torch.autograd.detect_anomaly(True):
                    # forward process
                    Matrix_PerturbData, NormalNoise, random_t_step, VecPerturbed_H = self.__PerturbHData(CleanHData, EXPAND_SCALE)
                    
                    # predict and compare with ground truth
                    loss = self.__CalculateLoss(OPT_PROJ_SE3_DENOISE_LOSS, model,
                                        random_t_step, NormalNoise, 
                                        Matrix_PerturbData, VecPerturbed_H)
                    train_loss = train_loss + loss
                    
                    # record loss in each epoch
                    #loss_trj = torch.cat((loss_trj, Loss.mean().detach()[None]), dim=0)

                    # clear gradient
                    for optim in optimizer:
                        optim.zero_grad()

                    # Backward pass calculate gradient
                    train_loss.backward(retain_graph=True)

                    # optimization by gradient
                    for optim in optimizer:
                        optim.step()

        #plt.plot(loss_trj)
        #plt.show()

## 1. Dataset #####################################################################
nTrainDataNum = RAW_DATA_NUM
RotTheta_data, TransData, HData = GetTrainData(nTrainDataNum)

# dataset, dataloader
TrainDataset = GraspDataset(HData)
Train_dataloader = DataLoader(TrainDataset, batch_size=1, shuffle=False) #shuffle = F, just for debug easily

## 2. set model ###################################################################
args_train = SetArgOfTrainModel()
model = load_pointcloud_grasp_diffusion(args_train)

## 3. train model (Forward process and learn denoise) #############################
device = torch.device('cpu')
context = PutShapeCodeMaskIntoModel(device, model)
Trainer = SimpleGraspTrainer(device)
Trainer.train(model, Train_dataloader)


## 4. sample based on trained model ###############################################