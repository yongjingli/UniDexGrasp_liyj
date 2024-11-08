# import os
# import shutil
#
# os.chdir('../')

from hydra import compose, initialize
import argparse
from network.trainer import Trainer
from network.train import process_config
from datasets.dex_dataset import DFCDataset   # 需要import Trainer才能import DFCDataset？
import torch
from torch.utils.data import Dataset

import numpy as np

import transforms3d
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes

import glob
import json

import sys
import os
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from network.models.loss import contact_map_of_m_to_n
from datasets.shadow_hand_builder import ShadowHandBuilder
import time
from tqdm import tqdm

from utils_data import save_2_ply


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="ipdf_config")
    parser.add_argument("--exp-dir", type=str, help="E.g., './ipdf_train'.")
    return parser.parse_args()


class DFCDatasetDebug(DFCDataset):
    def __init__(self, cfg, mode):
        super(DFCDatasetDebug, self).__init__(cfg, mode)

        self.fp = open("tmp.txt", "w")

        dataset_cfg = cfg["dataset"]
        if cfg["use_Shadow"]:
            self.hand_mesh_dir = pjoin(self.root_path, dataset_cfg["shadow_hand_mesh_dir"])
            self.hand_urdf_path = pjoin(self.root_path, dataset_cfg["shadow_urdf_path"])
            self.hand_builder = ShadowHandBuilder(self.hand_mesh_dir,
                                                  self.hand_urdf_path,
                                                  )
        else:
            # use Adroit
            # self.hand_mesh_dir = pjoin(self.root_path, dataset_cfg["adroit_hand_mesh_dir"])
            # self.hand_urdf_path = pjoin(self.root_path, dataset_cfg["adroit_urdf_path"])
            # self.hand_builder = AdroitHandBuilder(self.hand_mesh_dir,
            #                                       self.hand_urdf_path)
            raise NotImplementedError("Adroit is not supported yet")

    def __getitem__(self, item):
        item = 84172

        # time0 = time.time()
        file_path = self.file_list[item]  # e.g., "./data/DFCData/datasetv3.1/core/bottle-asdasfja12jaios9012/00000.npz"
        instance_no: str = file_path.split("/")[-2]
        category = file_path.split("/")[-3]  # e.g., core
        recorded_data = np.load(file_path, allow_pickle=True)

        qpos_dict = recorded_data["qpos"].item()
        global_translation = np.array([qpos_dict['WRJTx'], qpos_dict['WRJTy'], qpos_dict['WRJTz']])  # [3]
        global_rotation_mat = np.array(
            transforms3d.euler.euler2mat(qpos_dict['WRJRx'], qpos_dict['WRJRy'], qpos_dict['WRJRz']))  # [3, 3]
        object_scale = recorded_data["scale"]

        qpos = self.hand_builder.qpos_dict_to_qpos(qpos_dict)

        plane = recorded_data["plane"]
        obj_pc_path = pjoin(self.root_path, "DFCData", "meshes",
                            category, instance_no, "pcs_table.npy")
        pose_path = pjoin(self.root_path, "DFCData", "meshes",
                            category, instance_no, "poses.npy")
        pcs_table = torch.tensor(np.load(obj_pc_path, allow_pickle=True), dtype=torch.float)
        pose_matrices = torch.tensor(np.load(pose_path, allow_pickle=True), dtype=torch.float)
        index = (torch.tensor(plane[:3], dtype=torch.float) - pose_matrices[:, 2, :3]).norm(dim=1).argmin()
        pose_matrix = pose_matrices[index]
        pose_matrix[:2, 3] = 0
        pc = (pcs_table[index] @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]) / recorded_data ['scale'].item()

        object_pc = pc[:3000]
        table_pc = pc[3000:]

        max_diameter = 0.2
        n_samples_table_extra = 2000

        min_diameter = (object_pc[:, 0] ** 2 + object_pc[:, 1] ** 2).max()
        distances = min_diameter + (max_diameter - min_diameter) * torch.rand(n_samples_table_extra, dtype=torch.float) ** 0.5
        theta = 2 * np.pi * torch.rand(n_samples_table_extra, dtype=torch.float)
        table_pc_extra = torch.stack([distances * torch.cos(theta), distances * torch.sin(theta), torch.zeros_like(distances)], dim=1)
        table_pc = torch.cat([table_pc, table_pc_extra])
        table_pc_cropped = table_pc[table_pc[:, 0] ** 2 + table_pc[:, 1] ** 2 < max_diameter]
        table_pc_cropped_sampled = pytorch3d.ops.sample_farthest_points(table_pc_cropped.unsqueeze(0), K=1000)[0][0]
        object_pc = torch.cat([object_pc, table_pc_cropped_sampled])  # [N, 3]

        if self.dataset_cfg['fps']:
            obj_pc = pytorch3d.ops.sample_farthest_points(object_pc.unsqueeze(0), K=self.num_obj_points)[0][0]  # [NO, 3]
        else:
            obj_pc = object_pc

        global_rotation_mat = torch.from_numpy(global_rotation_mat).float()
        global_translation = torch.from_numpy(global_translation).float()

        # time1 = time.time() - time0
        # print("time1:", time1)

        if self.cfg["network_type"] == "ipdf":
            # plane_pose = plane2pose(plane)
            # global_rotation_mat = plane_pose[:3, :3] @ global_rotation_mat
            obj_gt_rotation = (pose_matrix[:3, :3] @ global_rotation_mat).T  # so that obj_gt_rotation @ obj_pc is what we want
            # place the table horizontally
            # obj_pc = obj_pc @ plane_pose[:3, :3].T + plane_pose[:3, 3]

            ret_dict = {
                "obj_pc": obj_pc,
                "obj_gt_rotation": obj_gt_rotation,
                "world_frame_hand_rotation_mat": global_rotation_mat,
            }
        elif self.cfg["network_type"] == "glow":  # TODO: 2: glow
            canon_obj_pc = torch.einsum('ba,cb,nc->na', global_rotation_mat, pose_matrix[:3, :3], obj_pc)
            hand_rotation_mat = np.eye(3)
            hand_translation = torch.einsum('a,ab->b', global_translation, global_rotation_mat)
            plane = torch.zeros_like(torch.from_numpy(plane))
            plane[..., 2] = 1
            ret_dict = {
                "obj_pc": obj_pc,
                "canon_obj_pc": canon_obj_pc,
                "hand_qpos": qpos,
                "canon_rotation": hand_rotation_mat,
                "canon_translation": hand_translation,
                "plane": plane
            }
        elif self.cfg["network_type"] == "cm_net":  # TODO: 2: Contact Map
            t_cm0 = time.time()
            # Canonicalize pc
            canon_obj_pc = torch.einsum('ba,cb,nc->na', global_rotation_mat, pose_matrix[:3, :3], obj_pc)
            # hand_rotation_mat = np.eye(3)
            hand_rotation_mat = torch.eye(3)
            hand_translation = torch.einsum('a,ab->b', global_translation, global_rotation_mat)

            # print(hand_rotation_mat, hand_translation, qpos)
            gt_hand_mesh = self.hand_builder.get_hand_mesh(hand_rotation_mat,
                                                             hand_translation,
                                                                 qpos=qpos)

            # import trimesh as tm
            # hand_mesh = tm.Trimesh(
            #     vertices=gt_hand_mesh.verts_list()[0].numpy(),
            #     faces=gt_hand_mesh.faces_list()[0].numpy()
            # )
            # hand_mesh.export("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/tmp.obj")

            gt_hand_mesh = gt_hand_mesh.cpu()
            # t = time.time()
            t_cm1 = time.time() - t_cm0


            gt_hand_pc = sample_points_from_meshes(
                gt_hand_mesh,
                num_samples=self.num_hand_points
            ).type(torch.float32).squeeze()  # torch.tensor: [NH, 3]
            contact_map = contact_map_of_m_to_n(canon_obj_pc, gt_hand_pc)  # [NO]

            # t = time.time()
            t_cm2 = time.time() - t_cm0
            ret_dict = {
                "canon_obj_pc": canon_obj_pc,
                "gt_hand_pc": gt_hand_pc,
                "contact_map": contact_map,
                "observed_hand_pc": gt_hand_pc
            }

            if self.dataset_cfg["perturb"]:
                pert_hand_translation = hand_translation + np.random.randn(3) * 0.03
                pert_qpos = qpos + np.random.randn(len(qpos)) * 0.1
                # pert_hand_mesh = self.hand_builder.get_hand_mesh(hand_rotation_mat,
                #                                                    pert_hand_translation,
                #                                                    qpos=pert_qpos)
                pert_hand_mesh = self.hand_builder.get_hand_mesh(hand_rotation_mat,
                                                                   pert_hand_translation,
                                                                   qpos=pert_qpos)
                pert_hand_pc = sample_points_from_meshes(
                    pert_hand_mesh,
                    num_samples=self.num_hand_points
                ).type(torch.float32).squeeze()  # torch.tensor: [NH, 3]
                ret_dict["observed_hand_pc"] = pert_hand_pc

            t_cm3 = time.time() - t_cm0

            txt = str(t_cm1) + " " + str(t_cm2) + " " + str(t_cm3) + " " + "\n"
            self.fp.write(txt)

            if t_cm3 > 0.1:
                print("t_cm:", t_cm1, t_cm2, t_cm3)
        else:
            raise NotImplementedError

        ret_dict["obj_scale"] = object_scale
        return ret_dict



def main(cfg):
    cfg = process_config(cfg)
    s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp"
    # train_loader = get_dex_dataloader(cfg, "train")
    # dataset = DFCDataset(cfg, "train")
    dataset = DFCDatasetDebug(cfg, "train")
    for data in tqdm(dataset, desc="dataset"):

        # cm-net
        canon_obj_pc = data["canon_obj_pc"]
        gt_hand_pc = data["gt_hand_pc"]
        contact_map = data["contact_map"]
        gt_hand_pc = data["gt_hand_pc"]
        observed_hand_pc = data["observed_hand_pc"]

        canon_obj_pc = canon_obj_pc.detach().cpu().numpy()
        gt_hand_pc = gt_hand_pc.detach().cpu().numpy()
        contact_map = contact_map.detach().cpu().numpy()
        observed_hand_pc = observed_hand_pc.detach().cpu().numpy()

        s_canon_obj_pc_path = os.path.join(s_root, "canon_obj_pc.ply")
        s_gt_hand_pc_path = os.path.join(s_root, "gt_hand_pc.ply")
        s_contact_map_path = os.path.join(s_root, "contact_map.ply")
        s_observed_hand_pc_path = os.path.join(s_root, "observed_hand_pc.ply")

        save_2_ply(s_canon_obj_pc_path, canon_obj_pc[:, 0], canon_obj_pc[:, 1], canon_obj_pc[:, 2], color=None)
        save_2_ply(s_gt_hand_pc_path, gt_hand_pc[:, 0], gt_hand_pc[:, 1], gt_hand_pc[:, 2], color=None)
        save_2_ply(s_observed_hand_pc_path, observed_hand_pc[:, 0], observed_hand_pc[:, 1], observed_hand_pc[:, 2], color=None)

        a = "fff"
        print("fff")
        exit(1)


def show_pt():
    import numpy as np
    import matplotlib.pyplot as plt

    pts0 = []
    pts1 = []
    pts2 = []
    with open("tmp.txt", "r") as fp:
        for line in fp.readlines():
            pt0 = float(line.strip().split(" ")[0])
            pt1 = float(line.strip().split(" ")[1])
            pt2 = float(line.strip().split(" ")[2])
            pts0.append(pt0)
            pts1.append(pt1)
            pts2.append(pt2)

    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(pts0)), pts0)

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(pts1)), pts1)

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(pts2)), pts2)
    plt.show()
    print(np.argmax(pts0), pts0[np.argmax(pts0)])


if __name__ == "__main__":
    args = parse_args()
    args.config_name = "cm_net_config"

    initialize(version_base=None, config_path="../configs", job_name="train")

    if args.exp_dir is None:
        cfg = compose(config_name=args.config_name)
    else:
        cfg = compose(config_name=args.config_name, overrides=[f"exp_dir={args.exp_dir}"])
    main(cfg)
    # show_pt()
