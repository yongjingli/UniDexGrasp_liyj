import os
os.chdir('../')

import cv2
import pytorch3d.ops

import shutil
import numpy as np
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
from utils_data import save_2_ply


def image_and_depth_to_point_cloud(image, depth, fx, fy, cx, cy, max_depth=5.0):
    rows, cols = depth.shape
    u, v = np.meshgrid(range(cols), range(rows))
    z = depth
    # 将深度为 0 或小于等于某个极小值的点标记为无效
    invalid_mask = np.bitwise_or(np.bitwise_or(z <= 0, z < np.finfo(np.float32).eps), z > max_depth)
    x = np.where(~invalid_mask, (u - cx) * z / fx, 0)
    y = np.where(~invalid_mask, (v - cy) * z / fy, 0)
    # z = z[~invalid_mask]
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = image.reshape(-1, 3)

    points = points[~invalid_mask.reshape(-1)]
    colors = colors[~invalid_mask.reshape(-1)]

    return points, colors


def get_object_surroud_points(scene_points, object_points, x_expand=0.05,
                              y_expand=0.05, z_expand=0.05):

    object_min_x = np.min(object_points[:, 0])
    object_max_x = np.max(object_points[:, 0])
    object_min_y = np.min(object_points[:, 1])
    object_max_y = np.max(object_points[:, 1])
    object_min_z = np.min(object_points[:, 2])
    object_max_z = np.max(object_points[:, 2])

    object_center_x = (object_min_x + object_max_x)/2
    object_center_y = (object_min_y + object_max_y)/2
    object_center_z = (object_min_z + object_max_z)/2
    # object_center = (object_center_x, object_center_y, object_center_z)
    object_center = (np.mean(object_points[:, 0]), np.mean(object_points[:, 1]), np.mean(object_points[:, 2]))

    x_mask = np.bitwise_and(scene_points[:, 0] > (object_min_x - x_expand),
                            scene_points[:, 0] < (object_max_x + x_expand))

    y_mask = np.bitwise_and(scene_points[:, 1] > (object_min_y - y_expand),
                            scene_points[:, 1] < (object_max_y + y_expand))

    z_mask = np.bitwise_and(scene_points[:, 2] > (object_min_z - z_expand),
                            scene_points[:, 2] < (object_max_z + z_expand))

    object_surroud_mask = np.bitwise_and(np.bitwise_and(x_mask, y_mask), z_mask)
    object_surroud_pts = scene_points[object_surroud_mask]
    return object_center, object_surroud_pts


def fps_pytorch_3d(pts, pt_num=1000):
    pts = torch.from_numpy(pts)
    pts_sampled = pytorch3d.ops.sample_farthest_points(pts.unsqueeze(0), K=pt_num)[0][0]
    # pts_sampled = pts_sampled.detach().cpu().numpy()
    return pts_sampled


def get_unidexgrasp_input(root, img_name, cam_k):
    s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp"
    # if os.path.exists(s_root):
    #     shutil.rmtree(s_root)
    # os.mkdir(s_root)

    img_root = os.path.join(root, "colors")
    # pose_root = os.path.join(root, "poses")
    mask_root = os.path.join(root, "masks_num")
    depth_root = os.path.join(root, "depths")

    img_path = os.path.join(img_root, img_name)
    mask_path = os.path.join(mask_root, img_name.replace("_color.jpg", "_mask.npy"))
    depth_path = os.path.join(depth_root, img_name.replace("_color.jpg", "_depth.npy"))

    img = cv2.imread(img_path)
    depth = np.load(depth_path)

    mask = np.load(mask_path)

    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask.astype(np.uint8), kernel)

    object_depth = np.zeros_like(depth)
    bg_depth = np.zeros_like(depth)

    object_depth[mask] = depth[mask]
    bg_depth[~mask] = depth[~mask]

    object_pts, obj_colors = image_and_depth_to_point_cloud(img, object_depth, fx=cam_k[0, 0], fy=cam_k[1, 1],
                                                            cx=cam_k[0, 2], cy=cam_k[1, 2], max_depth=5.0)

    bg_pts, bg_colors = image_and_depth_to_point_cloud(img, bg_depth, fx=cam_k[0, 0], fy=cam_k[1, 1],
                                                       cx=cam_k[0, 2], cy=cam_k[1, 2], max_depth=5.0)

    obj_center, bg_near_pts = get_object_surroud_points(bg_pts, object_pts, x_expand=0.05, y_expand=0.05, z_expand=0.05)

    object_pts = object_pts - obj_center
    bg_near_pts = bg_near_pts - obj_center

    object_pts = fps_pytorch_3d(object_pts, pt_num=3000)
    bg_near_pts = fps_pytorch_3d(bg_near_pts, pt_num=1000)

    # s_ply_path = os.path.join(s_root, "object.ply")
    # save_2_ply(s_ply_path, object_pts[:, 0], object_pts[:, 1], object_pts[:, 2])
    # exit(1)
    # save_2_ply(s_ply_path, object_pts[:, 0], object_pts[:, 1], object_pts[:, 2], obj_colors.tolist())
    #
    # s_ply_path = os.path.join(s_root, "bg.ply")
    # save_2_ply(s_ply_path, bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2])
    # # save_2_ply(s_ply_path, bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2], bg_colors.tolist())

    # s_ply_path = os.path.join(s_root, "bg_near.ply")
    # save_2_ply(s_ply_path, bg_near_pts[:, 0], bg_near_pts[:, 1], bg_near_pts[:, 2])

    # plt.subplot(3, 1, 1)
    # plt.imshow(img)
    #
    # plt.subplot(3, 1, 2)
    # plt.imshow(mask)
    #
    # plt.subplot(3, 1, 3)
    # # plt.imshow(np.clip(depth, 0, 2))
    # plt.imshow(object_depth)
    #
    # plt.show()
    #
    # print("fff")
    # exit(1)


    object_pc = torch.cat([object_pts, bg_near_pts])
    # object_pc = object_pts

    plane = torch.zeros(4)
    plane[2] = 1

    object_code = "no object code"

    scale = 1.0

    #  Sapien的点云的坐标系定义 x(forward), y(left), z(upward)
    trans_matrix = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    trans_matrix = torch.from_numpy(trans_matrix).float()

    object_pc_trans = object_pc @ trans_matrix

    ret_dict = {
        "object_code": object_code,
        # "obj_pc": object_pc,
        "obj_pc": object_pc_trans,
        "plane": plane,
        "scale": scale,
    }

    return ret_dict


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
    # final_results = []
    # for i in trange(len(result['hand_pose'])):
    #     final_results.append(eval_result(cfg['q1'], {k: result[k][i] for k in result.keys()}, hand_model, object_model, cfg['device']))
    # result.update(flatten_result(final_results))
    return result


def infer_unidexgrasp_realsense(cfg):

    cfg = process_config(cfg)
    logger = get_logger(cfg)

    # dict_keys(['rotation', 'pose'])
    rotation_trainer = get_trainer(cfg, 'rotation', logger)
    pose_trainer = get_trainer(cfg, 'pose', logger)
    # pose_trainer = get_trainer_joint(cfg, 'pose', logger)
    contact_net = get_contact_net(cfg)

    tta_loss = AdditionalLoss(cfg['tta'],
                              cfg['device'],
                              cfg['dataset']['num_obj_points'],
                              cfg['dataset']['num_hand_points'], contact_net)

    hand_model = tta_loss.hand_model

    # root_path = "/home/pxn-lyj/Egolee/data/unidexgrasp_data"
    root_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/generate_dataset"
    mesh_root = os.path.join(root_path, "DFCData", "meshes")

    object_model = KaolinModel(
        # data_root_path='data/DFCData/meshes',
        data_root_path=mesh_root,
        batch_size_each=1,
        device=cfg['device']
    )

    # object_list = get_mesh_data_object_list(root_path, 'test', scales=[0.06], n_samples=1)
    # object_list = get_mesh_data_object_list(root_path, 'test', scales=[1.0], n_samples=1)

    s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/infer_result"
    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)
    result = []

    root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/dexgrasp_show_realsense_20241009"
    img_root = os.path.join(root, "colors")

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    cam_k_path = os.path.join(root, "cam_k.txt")
    cam_k = np.loadtxt(cam_k_path)


    # torch.manual_seed(233)
    # torch.manual_seed(1024)
    torch.manual_seed(2048)

    for i, img_name in enumerate(img_names):
        # if img_name != "20_color.jpg":
        #     continue
        object_data = get_unidexgrasp_input(root, img_name, cam_k)
        batch_data = format_batch_data(object_data)
        for j in range(1):
            pred_dict, _ = rotation_trainer.test(batch_data)
            batch_data.update(pred_dict)

            pred_dict, _ = pose_trainer.test(batch_data)
            batch_data.update(pred_dict)

            points, hand_pose, plane_parameters = get_contact_net_input(batch_data)

            if 1:
                hand = tta_loss.hand_model(hand_pose, with_surface_points=True)
                discretized_cmap_pred = tta_loss.cmap_func(dict(canon_obj_pc=points, observed_hand_pc=hand['surface_points']))[
                    'contact_map'].exp()

                # 只将物体点云作为concat-map的输入
                # discretized_cmap_pred = tta_loss.cmap_func(dict(canon_obj_pc=points[:, :3000], observed_hand_pc=hand['surface_points']))[
                #     'contact_map'].exp()

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
            # pt_path = os.path.join(s_root, object_code_name + "_result.pt")
            pt_path = os.path.join(s_root, object_code_name + "_" + str(j) + "_result.pt")
            torch.save(infer_result, pt_path)

            # save concat-net
            # np.save(os.path.join(s_root, object_code_name + "_points.npy"), points.detach().cpu().numpy())
            # np.save(os.path.join(s_root, object_code_name + "_cm.npy"), cmap_pred.detach().cpu().numpy())
            exit(1)

        if i > 10:
            break

        print("end ")


if __name__ == "__main__":
    print("STart")
    args = parse_args()
    initialize(version_base=None, config_path="../configs", job_name="train")
    if args.exp_dir is None:
        cfg = compose(config_name=args.config_name)
    else:
        cfg = compose(config_name=args.config_name, overrides=[f"exp_dir={args.exp_dir}"])
    infer_unidexgrasp_realsense(cfg)
    print("End")