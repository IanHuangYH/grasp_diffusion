import os
import configargparse
import torch
from se3dif.utils import get_root_src, load_experiment_specifications, to_torch, to_numpy
from se3dif import datasets, losses, summaries, trainer
from torch.utils.data import DataLoader
from se3dif.models import loader

from se3dif.trainer.learning_rate_scheduler import get_learning_rate_schedules

from se3dif.datasets import AcronymGraspsDirectory
from mesh_to_sdf.scan import ScanPointcloud
import numpy as np
import scipy.spatial.transform

from se3dif.samplers import Grasp_AnnealedLD

base_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.dirname(__file__ + '/../../../../../'))


def parse_train_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--specs_file_dir', type=str, default=os.path.join(base_dir, 'params')
                   , help='root for saving logging')

    p.add_argument('--spec_file', type=str, default='multiobject_partialp_graspdif'
                   , help='root for saving logging')

    p.add_argument('--summary', type=bool, default=True
                   , help='activate or deactivate summary')

    p.add_argument('--saving_root', type=str, default=os.path.join(get_root_src(), 'logs')
                   , help='root for saving logging')

    p.add_argument('--models_root', type=str, default=root_dir
                   , help='root for saving logging')

    p.add_argument('--device',  type=str, default='cuda',)
    p.add_argument('--class_type', type=str, default='Mug')

    opt = p.parse_args()
    return opt

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
    P += -P_mean

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

def main(opt):

    ## Load training args ##
    #Ian# 
    # opt.specs_file_dir: '/code/grasp_diffusion/scripts/train/params'
    # opt.spec_file: 'multiobject_partialp_graspdif'
    spec_file = os.path.join(opt.specs_file_dir, opt.spec_file)
    args = load_experiment_specifications(spec_file)

    # saving directories
    #Ian#
    root_dir = opt.saving_root # opt.saving_root: '/code/logs'
    exp_dir  = os.path.join(root_dir, args['exp_log_dir']) # exp_dir: '/code/logs/multiobject_partial_graspdif'
    args['saving_folder'] = exp_dir


    if opt.device =='cuda':
        if 'cuda_device' in args:
            cuda_device = args['cuda_device']
        else:
            cuda_device = 0
        device = torch.device('cuda:' + str(cuda_device) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    ## Dataset
    train_dataset = datasets.PartialPointcloudAcronymAndSDFDataset(augmented_rotation=True, one_object=args['single_object'])
    train_dataloader = DataLoader(train_dataset, batch_size=args['TrainSpecs']['batch_size'], shuffle=True, drop_last=True)
    test_dataset = datasets.PartialPointcloudAcronymAndSDFDataset(augmented_rotation=True, one_object=args['single_object'],
                                                                  test_files=train_dataset.test_grasp_files)
    test_dataloader = DataLoader(test_dataset, batch_size=args['TrainSpecs']['batch_size'], shuffle=True, drop_last=True)
    
    ## Model
    args['device'] = device
    #Ian add argument for pre-trained model
    #args['pretrained_model'] = 'partial_grasp_dif'
    model = loader.load_model(args)

    if 'pretrained_model' not in args: # need training
    #if (True): # need training
        # Losses
        loss = losses.get_losses(args)
        loss_fn = val_loss_fn = loss.loss_fn

        ## Summaries
        summary = summaries.get_summary(args, opt.summary)

        ## Optimizer
        lr_schedules = get_learning_rate_schedules(args)
        optimizer = torch.optim.Adam([
                {
                    "params": model.vision_encoder.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
                {
                    "params": model.feature_encoder.parameters(),
                    "lr": lr_schedules[1].get_learning_rate(0),
                },
                {
                    "params": model.decoder.parameters(),
                    "lr": lr_schedules[2].get_learning_rate(0),
                },
            ])
        
        # Train
        trainer.train(model=model.float(), train_dataloader=train_dataloader, epochs=args['TrainSpecs']['num_epochs'], model_dir= exp_dir,
                    summary_fn=summary, device=device, lr=1e-4, optimizers=[optimizer],
                    steps_til_summary=args['TrainSpecs']['steps_til_summary'],
                    epochs_til_checkpoint=args['TrainSpecs']['epochs_til_checkpoint'],
                    loss_fn=loss_fn, iters_til_checkpoint=args['TrainSpecs']['iters_til_checkpoint'],
                    clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True,
                    val_dataloader=test_dataloader)
        
    #parse argument for sampling
    args_sample = parse_Sample_args()

    print('##########################################################')
    print('Object Class: {}'.format(args_sample.obj_class))
    print(args_sample.obj_id)
    print('##########################################################')

    n_grasps = int(args_sample.n_grasps)
    obj_id = int(args_sample.obj_id)
    obj_class = args_sample.obj_class
    n_envs = 30

    #set model
    P, mesh = sample_pointcloud(obj_id, obj_class)
    context = to_torch(P[None,...], device)
    model.set_latent(context, batch=10)

    # set sampling method
    generator = Grasp_AnnealedLD(model, batch=10, T = 70, T_fit=50, k_steps=1, device=device)
    
    H = generator.sample()
    H[..., :3, -1] *=1/8.

    ## Visualize results ##
    from se3dif.visualization import grasp_visualization

    vis_H = H.squeeze()
    P *=1/8
    mesh = mesh.apply_scale(1/8)
    grasp_visualization.visualize_grasps(to_numpy(H), p_cloud=P)#, mesh=mesh)


if __name__ == '__main__':
    args = parse_train_args()
    main(args)