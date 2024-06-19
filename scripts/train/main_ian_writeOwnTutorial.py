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
from tqdm.autonotebook import tqdm
import os

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
#from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D

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

def GenerateShapeCodeMask(device):
    args_sample = parse_Sample_args()
    obj_id = int(args_sample.obj_id)
    obj_class = args_sample.obj_class
    P, mesh = sample_pointcloud(obj_id, obj_class)
    context = to_torch(P[None,...], device)
    #model.set_latent(context, batch=EXPAND_SCALE * BATCH_SIZE)
    return context

def marginal_prob_std_np(t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

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
    
    def __PerturbHData(self, Data:torch.tensor, ExpandNum: int, context, eps=1e-5) -> torch.tensor:
        '''Data: H with 4x4 tensor'''
        # !!! must add, or torch cannot teack the gradient
        model.set_latent(context, batch=EXPAND_SCALE * BATCH_SIZE)

        ## 1. expand H ##
        HSingle = SO3_R3(R=Data[...,:3, :3], t=Data[...,:3, -1])
        Matrix_expand_H = HSingle.to_matrix().repeat(ExpandNum,1,1)
        H_expand = SO3_R3(R=Matrix_expand_H[...,:3, :3], t=Matrix_expand_H[...,:3, -1])
        
        ## 2. H to vector ##
        VecCleanH = H_expand.log_map()
        
        ## 3. generate noise
        random_t_step = torch.rand_like(VecCleanH[...,0], device=self.m_device) * (1. - eps) + eps
        VecNoise = torch.randn_like(VecCleanH)
        std = self.marginal_prob_std(random_t_step)
        VecPerturbed_H = VecCleanH + VecNoise * std[..., None]
        VecPerturbed_H = VecPerturbed_H.detach()
        VecPerturbed_H.requires_grad_(True)

        with torch.set_grad_enabled(True):
            Matrix_PerturbData = SO3_R3().exp_map(VecPerturbed_H).to_matrix()

        ## 4. normalize generate noise
        NormalNoise = VecNoise/std[...,None]


        return Matrix_PerturbData, NormalNoise, random_t_step, VecPerturbed_H
    
    def __CalculateLoss(self, option:int, model, 
                      random_t_step: torch.tensor, NormalNoise:torch.tensor, 
                      Matrix_PerturbData:torch.tensor, VecPerturbed_H:torch.tensor):

        with torch.set_grad_enabled(True):
            energy = model(Matrix_PerturbData, random_t_step)
            grad_energy = torch.autograd.grad(energy.sum(), VecPerturbed_H,
                                            only_inputs=True, retain_graph=True, create_graph=True)[0]
        loss_fn = torch.nn.L1Loss()
        loss = loss_fn(grad_energy, NormalNoise)/10.

        info = {'E_theta': grad_energy}
        loss_dict = {"Log_q": loss}
            
        return loss_dict, info
    
    def loss_fn_ian(self, model, Data:torch.tensor, ExpandNum, context, eps=1e-5):
        model.set_latent(context, batch=EXPAND_SCALE * BATCH_SIZE)
        ## 1. expand H ##
        HSingle = SO3_R3(R=Data[...,:3, :3], t=Data[...,:3, -1])
        Matrix_expand_H = HSingle.to_matrix().repeat(ExpandNum,1,1)
        H_expand = SO3_R3(R=Matrix_expand_H[...,:3, :3], t=Matrix_expand_H[...,:3, -1])
        # Data : torch.tensor = Data.reshape(-1, 4, 4)
        # H_expand : SO3_R3 = SO3_R3(R=Data[...,:3, :3], t=Data[...,:3, -1])
        
        ## 2. H to vector ##
        VecCleanH : torch.tensor = H_expand.log_map()
        
        ## 3. generate noise
        random_t_step : torch.tensor = torch.rand_like(VecCleanH[...,0], device=self.m_device) * (1. - eps) + eps
        VecNoise : torch.tensor = torch.randn_like(VecCleanH)
        std = self.marginal_prob_std(random_t_step)
        VecPerturbed_H : torch.tensor = VecCleanH + VecNoise * std[..., None]
        VecPerturbed_H = VecPerturbed_H.detach()
        VecPerturbed_H.requires_grad_(True)

        with torch.set_grad_enabled(True):
            Matrix_PerturbData = SO3_R3().exp_map(VecPerturbed_H).to_matrix()
            energy = model(Matrix_PerturbData, random_t_step)
            grad_energy = torch.autograd.grad(energy.sum(), VecPerturbed_H,
                                            only_inputs=True, retain_graph=True, create_graph=True)[0]


        # 4. normalize generate noise
        NormalNoise = VecNoise/std[...,None]
        loss_fn = torch.nn.L1Loss()
        loss = loss_fn(grad_energy, NormalNoise)/10.

        info = {'denoise': grad_energy}
        loss_dict = {"Score loss": loss}
        return loss_dict, info

    def train(self, model, TrainDataLoader: DataLoader, context, MODEL_LOAD_PATH:str):
        optimizer = [self.__SetOptimizer()]
        Train_Iter = TRAIN_EPOCH
        loss_trj = torch.zeros(0)

        with tqdm(total=len(TrainDataLoader) * Train_Iter) as pbar:
            train_losses_save = []
            for idx_Data, CleanHData in enumerate(TrainDataLoader):
                for Iter in range(Train_Iter): 
                
                    # losses, iter_info = self.loss_fn_ian(model, CleanHData, EXPAND_SCALE, context)
                    # forward process
                    Matrix_PerturbData, NormalNoise, random_t_step, VecPerturbed_H = self.__PerturbHData(CleanHData, EXPAND_SCALE, context)
                    
                    # predict and compare with ground truth
                    losses, iter_info = self.__CalculateLoss(OPT_PROJ_SE3_DENOISE_LOSS, model,
                                                            random_t_step, NormalNoise, 
                                                            Matrix_PerturbData, VecPerturbed_H)
                    
                    train_loss = 0.

                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        train_loss += single_loss
                    
                    train_losses_save.append(train_loss.item())
                    # record loss in each epoch
                    loss_trj = torch.cat((loss_trj, train_loss.detach()[None]), dim=0)

                    # clear gradient
                    for optim in optimizer:
                        optim.zero_grad()

                    # Backward pass calculate gradient
                    train_loss.backward()

                    # optimization by gradient
                    for optim in optimizer:
                        optim.step()

                    pbar.update(1)
        
        torch.save(model.state_dict(),
                   os.path.join(MODEL_LOAD_PATH, 'model_ian_tutorial.pth'))
        np.savetxt(os.path.join(MODEL_LOAD_PATH, 'train_losses_ian_tutorial.txt'),
                   np.array(train_losses_save))            

        #plt.plot(loss_trj)
        #plt.show()

def LangevinSampleStep(H0_nosie: torch.tensor, model, device, TotalStep:int):
    eps = 1e-3
    alpha = 1e-3
    VecH0 = SO3_R3(R=H0_nosie[..., :3, :3] , t=H0_nosie[..., :3, -1]).log_map()
    H_traj = []
    for t in range(TotalStep):
        ## coefficient
        k = ((TotalStep - t)/TotalStep) + eps
        sigma_T = marginal_prob_std_np(eps)
        sigma_i = marginal_prob_std_np(k)
        ratio = sigma_i/sigma_T
        c_lr = alpha*ratio**2 #alpha_k^2 in (7)

        ## 1. add random noise in sample
        noise = torch.randn_like(VecH0)
        noise = np.sqrt(c_lr)*noise
        VecH0_AddNoise = VecH0 + np.sqrt(alpha)*ratio*noise


        ## 2. Compute gradient ##
        InputStep = k*torch.ones_like(VecH0_AddNoise[...,0])
        VecH0_AddNoise = VecH0_AddNoise.detach().requires_grad_(True)
        H0_AddNoise = SO3_R3().exp_map(VecH0_AddNoise).to_matrix()
        energy = model(H0_AddNoise, InputStep)
        grad_energy = torch.autograd.grad(energy.sum(), VecH0_AddNoise, only_inputs=True)[0]

        ## 3. Evolve gradient ##
        delta_x = -.5*c_lr*grad_energy
        VecUpdate = VecH0_AddNoise + delta_x

        ## Build H for output##
        H_update : SO3_R3 = SO3_R3().exp_map(VecUpdate).to_matrix().reshape(4,4)
        H_traj.append(H_update)

        ## Update ##
        VecH0 = VecUpdate
    return VecUpdate, H_update, H_traj
        
class GraspingDrawer():
    def __init__(self) -> None:
        self.m_arrowLength = 0.1
        self.fig2D = plt.figure()
        self.ax2D = self.fig2D.gca()
        self.fig3D = plt.figure()
        self.ax3D = self.fig3D.gca(projection='3d')
        


    def DrawSingleH(self, H:torch.tensor, in_color='b') -> None:
        # basePt = np.array([H[0][3]], [H[1][3]], [H[2][3]])
        # baseVec = np.array([self.m_arrowLength], [0], [0])
        basePt : np.array = H[:3, -1].detach().numpy().reshape(3,1)
        baseVec : np.array = np.array([[self.m_arrowLength], [0.], [0.]])
        RotateMat : np.array = H[:3, :3].detach().numpy().reshape(3,3)
        RotateVec = np.matmul(RotateMat, baseVec)

        TarPt = basePt + RotateVec
        PtX, PtY, PtZ = [basePt[0][0], TarPt[0][0]], [basePt[1][0], TarPt[1][0]], [basePt[2][0], TarPt[2][0]]
        #plt.plot(PtX, PtY, PtZ, marker = 'o', color=in_color)
        self.ax2D.plot(PtX, PtY, marker = 'o', color=in_color)
        self.ax3D.plot(PtX, PtY, PtZ, marker = 'o', color=in_color)
        plt.show()
        

    def DrawDataSet(self, HDataSet: torch.tensor, in_color='b')-> None:
        for H in HDataSet:
            self.DrawSingleH(H, in_color)
##########################################################################################################
## 0. Basic setting ###############################################################
#argument for training
OPT_PROJ_SE3_DENOISE_LOSS = 1
RAW_DATA_NUM = 4
EXPAND_SCALE = 400
TRAIN_EPOCH = 1000
BATCH_SIZE = 1

#argument for sampling
PRE_TRAIN_MODEL = 1
device = torch.device('cpu')
MODEL_LOAD_PATH = '/code/result/'
PRE_TRAIN_MODEL_NAME = 'model_ian_tutorial.pth'
SAMPLE_NUM = 1
SAMPLE_STEP = 500




# 1. Dataset #####################################################################
nTrainDataNum = RAW_DATA_NUM
RotTheta_data, TransData, HData = GetTrainData(nTrainDataNum)

# dataset, dataloader
TrainDataset = GraspDataset(HData)
Train_dataloader = DataLoader(TrainDataset, batch_size=1, shuffle=False) #shuffle = F, just for debug easily

## 2. set model ###################################################################
args_train = SetArgOfTrainModel()
model = load_pointcloud_grasp_diffusion(args_train)

## 3. train model (Forward process and learn denoise) #############################
context = GenerateShapeCodeMask(device)
if PRE_TRAIN_MODEL == 0: # train own model
    Trainer = SimpleGraspTrainer(device)
    Trainer.train(model, Train_dataloader, context, MODEL_LOAD_PATH)
elif PRE_TRAIN_MODEL == 1: # use checkpoint
    model_path = os.path.join(MODEL_LOAD_PATH, PRE_TRAIN_MODEL_NAME) # find the checkpoint file .pth
    model.load_state_dict(torch.load(model_path, map_location=device))
else: # wrong
    print("Wrong Flag!")


## 4. sample based on trained model ##################################################
model.set_latent(context, batch=SAMPLE_NUM * BATCH_SIZE)
# random gaussian H
H0_nosie = SO3_R3().sample(SAMPLE_NUM).to(device, torch.float32) # SAMPLE_NUM x 4 x 4 
# R0_noise : torch.tensor = SO3.rand(SAMPLE_NUM).to_matrix() 
# x0_noise : torch.tensor = torch.randn(SAMPLE_NUM, 3) # SAMPLE_NUM x 3
VecFinal, H_Final, H_traj = LangevinSampleStep(H0_nosie, model, device, SAMPLE_STEP)

## 5. plot ###########################################################################
Drawer = GraspingDrawer()
Drawer.DrawDataSet(HData)
Drawer.DrawDataSet(H_traj,'r')
#Drawer.DrawSingleH(H_Final,'r')
print("finish")

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # Make the grid
# x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
#                       np.arange(-0.8, 1, 0.2),
#                       np.arange(-0.8, 1, 0.8))

# # Make the direction data for the arrows
# u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
# v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
# w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
#      np.sin(np.pi * z))

# ax.quiver(0, 0, 0, 1, 1, 0, length=0.1, normalize=True)

# plt.show()