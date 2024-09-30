cfg = {'network_type': 'ipdf', 'use_DFCData': True, 'use_Shadow': True,
       'exp_dir': './runs/exp_ipdf', 'wandb_offline': True, 'wandb_debug_mode': True,
       'freq': {'step_epoch': 100, 'save': 10000, 'plot': 100, 'test': 75000}, 'optimizer': 'Adam',
       'weight_decay': 0.0001, 'learning_rate': 0.001, 'lr_policy': 'step', 'lr_gamma': 0.5, 'lr_step_size': 40,
       'lr_clip': 1e-05, 'momentum_original': 0.1, 'momentum_decay': 0.5, 'momentum_step_size': 20, 'momentum_min': 0.01,
       'weight_init': 'xavier', 'total_epoch': 250, 'resume_epoch': -1, 'cuda_id': 0, 'num_workers': 1, 'batch_size': 16,
       'model': {'network': {'type': 'pn_rotation_net'}, 'number_fourier_components': 1, 'mlp_layer_sizes': [256, 256, 256],
                 'num_train_queries': 4096, 'loss_weight': {'nll': 1.0}},
       'dataset': {'root_path': 'data', 'dataset_dir': 'DFCData', 'hand_global_trans': [0, -0.7, 0.2],
                   'hand_global_rotation_xyz': [-1.57, 0, 3.14], 'num_obj_points': 1024, 'num_hand_points': 1024,
                   'shadow_hand_mesh_dir': 'mjcf/meshes', 'shadow_urdf_path': 'mjcf/shadow_hand.xml', 'fps': True},
       'device': 'cuda:0'}

from network.models.graspipdf.ipdf_network import PointNetPP
import numpy as np
import torch
from network.trainer import Trainer
from omegaconf.omegaconf import open_dict
from hydra import compose, initialize
import os
import logging


def debug_pointnet_pp(cfg):
    # device = cfg.device
    device = cfg["device"]
    in_tensor = np.load("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/obj_pc.npy")
    in_tensor = torch.from_numpy(in_tensor).to(device)

    backbone = PointNetPP(cfg).to(device)

    out = backbone(in_tensor)
    # print(out)
    print(backbone.backbone.lin1.bias)


def debug_ipdf_trainer(net_cfg):
    """ Logging """
    log_dir = cfg["exp_dir"]
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("EvalModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    initialize(version_base=None, config_path="../configs", job_name="train")
    net_cfg = compose(f"ipdf_config")
    with open_dict(net_cfg):
        net_cfg['device'] = cfg['device']

    # logger = None
    trainer = Trainer(net_cfg, logger)
    trainer.resume()

    trainer.model.eval()
    # print(trainer.model.net.backbone.backbone.lin1.weight)

    in_tensor = np.load("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/obj_pc.npy")
    in_tensor = torch.from_numpy(in_tensor).to(cfg["device"])
    in_tensor = in_tensor[0:1, ]
    feat = trainer.model.net.backbone(in_tensor).reshape(len(in_tensor), -1)
    print(feat.shape, feat)

    inputs = {}
    inputs["obj_pc"] = in_tensor
    sampled_rotations = trainer.model.net.sample_rotations(inputs)
    print(sampled_rotations)



if __name__ == "__main__":
    print("STart")
    # debug_pointnet_pp(cfg)
    debug_ipdf_trainer(cfg)
    print("End")
