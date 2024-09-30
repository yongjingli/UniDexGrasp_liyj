import os
import shutil

os.chdir('../')

import logging
import torch
import argparse
from tqdm import tqdm, trange
from utils.global_utils import result_to_loader, flatten_result
from utils.eval_utils import KaolinModel, eval_result
from collections import OrderedDict
from hydra import compose, initialize
from omegaconf.omegaconf import open_dict
from os.path import join as pjoin
from omegaconf import OmegaConf
from network.trainer import Trainer
from network.train import process_config
from network.models.contactnet.contact_network import ContactMapNet
from network.models.model import get_model
from utils_data import get_mesh_data_object_list, get_object_data, format_batch_data
from utils.hand_model import AdditionalLoss, add_rotation_to_hand_pose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="eval_config")
    parser.add_argument("--exp-dir", type=str, help="E.g., './infer_result'.")
    return parser.parse_args()


def get_logger(cfg):
    log_dir = cfg["exp_dir"]
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("EvalModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_trainer(cfg, key, logger):
    net_cfg = compose(f"{cfg['models'][key]['type']}_config")
    with open_dict(net_cfg):
        net_cfg['device'] = cfg['device']
    trainer = Trainer(net_cfg, logger)
    trainer.resume()
    return trainer


def get_trainer_joint(cfg, key, logger):

    net_cfg = compose(f"{cfg['models'][key]['type']}_joint_config")

    with open_dict(net_cfg):
        net_cfg['device'] = cfg['device']
    trainer = Trainer(net_cfg, logger)
    trainer.resume()
    return trainer

def get_contact_net(cfg):
    contact_cfg = compose(f"{cfg['tta']['contact_net']['type']}_config")
    with open_dict(contact_cfg):
        contact_cfg['device'] = cfg['device']
    contact_net = ContactMapNet(contact_cfg)
    ckpt_dir = pjoin(contact_cfg['exp_dir'], 'ckpt')
    model_name = get_model(ckpt_dir, contact_cfg.get('resume_epoch', None))
    ckpt = torch.load(model_name)['model']
    new_ckpt = OrderedDict()
    for name in ckpt.keys():
        new_name = name.replace('net.', '')
        if new_name.startswith('backbone.'):
            new_name = new_name.replace('backbone.', '')
        new_ckpt[new_name] = ckpt[name]

    contact_net.load_state_dict(new_ckpt)
    contact_net = contact_net.to(cfg['device'])
    contact_net.eval()
    return contact_net


def get_contact_net_input(data):
    points = data['canon_obj_pc'].cuda()
    hand_pose = torch.cat([data['canon_translation'], torch.zeros_like(data['canon_translation']), data['hand_qpos']],
                          dim=-1)
    old_hand_pose = hand_pose.clone()
    data['hand_pose'] = add_rotation_to_hand_pose(old_hand_pose, data['sampled_rotation'])
    plane_parameters = data['canon_plane'].cuda()
    hand_pose = hand_pose.cuda()
    return points, hand_pose, plane_parameters


def format_result(result, cfg, hand_model, object_model):
    result = flatten_result(result)
    final_results = []
    for i in trange(len(result['hand_pose'])):
        final_results.append(eval_result(cfg['q1'], {k: result[k][i] for k in result.keys()}, hand_model, object_model, cfg['device']))
    result.update(flatten_result(final_results))
    return result


def main(cfg):
    cfg = process_config(cfg)
    logger = get_logger(cfg)

    # dict_keys(['rotation', 'pose'])
    rotation_trainer = get_trainer(cfg, 'rotation', logger)
    # pose_trainer = get_trainer(cfg, 'pose', logger)
    pose_trainer = get_trainer_joint(cfg, 'pose', logger)
    contact_net = get_contact_net(cfg)

    tta_loss = AdditionalLoss(cfg['tta'],
                              cfg['device'],
                              cfg['dataset']['num_obj_points'],
                              cfg['dataset']['num_hand_points'], contact_net)

    hand_model = tta_loss.hand_model
    object_model = KaolinModel(
        data_root_path='data/DFCData/meshes',
        batch_size_each=1,
        device=cfg['device']
    )

    root_path = "/home/pxn-lyj/Egolee/data/unidexgrasp_data"
    object_list = get_mesh_data_object_list(root_path, 'test', scales=[0.06], n_samples=1)
    result = []

    # s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/eval"
    s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/infer_result"
    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)
    torch.manual_seed(233)
    # torch.manual_seed(1024)
    # torch.manual_seed(2048)

    for i in range(len(object_list)):
        object_data = get_object_data(object_list[i], use_table_pc_extra=False)
        batch_data = format_batch_data(object_data)

        pred_dict, _ = rotation_trainer.test(batch_data)
        batch_data.update(pred_dict)

        pred_dict, _ = pose_trainer.test(batch_data)
        batch_data.update(pred_dict)

        points, hand_pose, plane_parameters = get_contact_net_input(batch_data)

        if 0:
            hand = tta_loss.hand_model(hand_pose, with_surface_points=True)
            discretized_cmap_pred = tta_loss.cmap_func(dict(canon_obj_pc=points, observed_hand_pc=hand['surface_points']))[
                'contact_map'].exp()
            cmap_pred = (torch.argmax(discretized_cmap_pred, dim=-1) + 0.5) / discretized_cmap_pred.shape[-1]

            hand_pose.requires_grad_()
            optimizer = torch.optim.Adam([hand_pose], lr=cfg['tta']['lr'])
            for t in range(cfg['tta']['iterations']):
                optimizer.zero_grad()
                loss = tta_loss.tta_loss(hand_pose, points, cmap_pred, plane_parameters)
                loss.backward()
                optimizer.step()

        batch_data['tta_hand_pose'] = add_rotation_to_hand_pose(hand_pose.detach().cpu(), batch_data['sampled_rotation'].detach().cpu())

        infer_result = {k: v.cpu() if type(v) == torch.Tensor else v for k, v in batch_data.items()}
        result.append(infer_result)

        infer_result = format_result([infer_result], cfg, hand_model, object_model)

        object_code_name = batch_data['object_code'][0].replace("/", "_")
        pt_path = os.path.join(s_root, object_code_name + "_result.pt")
        torch.save(infer_result, pt_path)

        if i > 10:
            break

    # result = format_result(result, cfg, hand_model, object_model)
    # pt_path = s_root + '/result.pt'
    # torch.save(result, pt_path)


if __name__ == "__main__":
    args = parse_args()
    initialize(version_base=None, config_path="../configs", job_name="train")
    if args.exp_dir is None:
        cfg = compose(config_name=args.config_name)
    else:
        cfg = compose(config_name=args.config_name, overrides=[f"exp_dir={args.exp_dir}"])
    main(cfg)