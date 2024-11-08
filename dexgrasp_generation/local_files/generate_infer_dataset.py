import os

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import transforms3d
import torch
import pytorch3d.io
import pytorch3d.ops
import pytorch3d.structures
import sapien.core as sapien
from multiprocessing import Pool, current_process
from tqdm import tqdm


from scripts.generate_object_pc2 import sample_projected
from scripts.generate_object_pose2 import generate_object_pose
from scripts.generate_object_table_pc2 import sample_projected as sample_projected_table


def generate_object_pc(meshes_root):
    parser = argparse.ArgumentParser()
    # experiments settings
    parser.add_argument('--data_root_path', type=str, default='../data/DFCData/meshes')
    parser.add_argument('--n_poses', type=int, default=100)
    parser.add_argument('--max_n_points', type=int, default=9000)
    parser.add_argument('--num_samples', type=int, default=3000)
    parser.add_argument('--n_cpu', type=int, default=8)
    # parser.add_argument('--n_cameras', type=int, default=6)
    # parser.add_argument('--theta', type=float, default=np.pi / 4)
    # parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--gpu_list', type=str, nargs='*', default=['0', '1', '2', '3'])
    # camera settings
    parser.add_argument('--camera_distance', type=float, default=0.5)
    parser.add_argument('--camera_height', type=float, default=0.05)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--near', type=float, default=0.1)
    parser.add_argument('--far', type=float, default=100)
    args = parser.parse_args()

    args.data_root_path = meshes_root

    object_category_list = os.listdir(args.data_root_path)
    object_code_list = []
    for object_category in object_category_list:
        object_code_list += [os.path.join(object_category, object_code) for object_code in
                             sorted(os.listdir(os.path.join(args.data_root_path, object_category)))]
    # object_code_list = [object_code for object_code in object_code_list if not os.path.exists(os.path.join(args.data_root_path, object_code, 'pcs.npy'))]

    # object_code_list = object_code_list[:1]

    parameters = []
    for idx, object_code in enumerate(object_code_list):
        parameters.append((args, object_code, idx))

    with Pool(args.n_cpu) as p:
        it = tqdm(p.imap(sample_projected, parameters), desc='sampling', total=len(parameters))
        list(it)


def generate_object_poses(meshes_root):
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--data_root_path', type=str, default='../data/DFCData/meshes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--overwrite', type=bool, default=True)
    parser.add_argument('--n_cpu', type=int, default=16)
    # simulator settings
    parser.add_argument('--sim_steps', type=int, default=1000)
    parser.add_argument('--time_step', type=float, default=1 / 100)
    parser.add_argument('--restitution', type=float, default=0.01)

    args = parser.parse_args()
    args.data_root_path = meshes_root

    # seed
    np.random.seed(args.seed)

    # # load object list
    # object_code_list = os.listdir(args.data_root_path)
    #
    # # generate object pose
    # # for object_code in tqdm(object_code_list, desc='generating'):
    # #     generate_object_pose((args, object_code))
    # with Pool(args.n_cpu) as p:
    #     param_list = []
    #     for object_code in object_code_list:
    #         param_list.append((args, object_code))
    #     list(tqdm(p.imap(generate_object_pose, param_list), desc='generating', total=len(param_list), miniters=1))


    object_category_list = os.listdir(args.data_root_path)
    object_code_list = []
    for object_category in object_category_list:
        object_code_list += [os.path.join(object_category, object_code) for object_code in sorted(os.listdir(os.path.join(args.data_root_path, object_category)))]

    parameters = []
    for idx, object_code in enumerate(object_code_list):
        # parameters.append((args, object_code, idx))
        parameters.append((args, object_code))

    # generate object pose
    # for object_code in tqdm(object_code_list, desc='generating'):
    #     generate_object_pose((args, object_code))
    with Pool(args.n_cpu) as p:
        # param_list = []
        # for object_code in object_code_list:
        #     param_list.append((args, object_code))
        list(tqdm(p.imap(generate_object_pose, parameters), desc='generating', total=len(parameters), miniters=1))



def generate_object_table_pc(meshes_root):
    parser = argparse.ArgumentParser()
    # experiments settings
    parser.add_argument('--data_root_path', type=str, default='data/DFCData/meshes')
    parser.add_argument('--n_poses', type=int, default=100)
    parser.add_argument('--max_n_points', type=int, default=9000)
    parser.add_argument('--num_samples', type=int, default=3000)
    parser.add_argument('--table_distance', type=float, default=0.2)
    parser.add_argument('--max_n_points_table', type=int, default=3000)
    parser.add_argument('--num_samples_table', type=int, default=1000)
    parser.add_argument('--n_cpu', type=int, default=8)
    # parser.add_argument('--n_cameras', type=int, default=6)
    # parser.add_argument('--theta', type=float, default=np.pi / 4)
    # parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--gpu_list', type=str, nargs='*', default=['0', '1', '2', '3'])
    # camera settings
    parser.add_argument('--camera_distance', type=float, default=0.5)
    parser.add_argument('--camera_height', type=float, default=0.05)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--near', type=float, default=0.1)
    parser.add_argument('--far', type=float, default=100)
    args = parser.parse_args()
    args.data_root_path = meshes_root

    object_category_list = os.listdir(args.data_root_path)
    object_code_list = []
    for object_category in object_category_list:
        object_code_list += [os.path.join(object_category, object_code) for object_code in
                             sorted(os.listdir(os.path.join(args.data_root_path, object_category)))]
    # object_code_list = [object_code for object_code in object_code_list if not os.path.exists(os.path.join(args.data_root_path, object_code, 'pcs.npy'))]

    # object_code_list = object_code_list[:1]

    parameters = []
    for idx, object_code in enumerate(object_code_list):
        parameters.append((args, object_code, idx))

    with Pool(args.n_cpu) as p:
        it = tqdm(p.imap(sample_projected_table, parameters), desc='sampling', total=len(parameters))
        list(it)


def write_splits(meshes_root):
    splits_root = meshes_root.replace("meshes", "splits")
    splits_names = os.listdir(meshes_root)
    for splits_name in splits_names:
        s_splits_path = os.path.join(splits_root, splits_name + ".json")
        infos = {}

        object_names = os.listdir(os.path.join(meshes_root, splits_name))
        infos["train"] = object_names
        infos["test"] = object_names

        json_string = json.dumps(infos)
        with open(s_splits_path, "w") as file:
            file.write(json_string)



if __name__ == "__main__":
    print("STart")
    root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/generate_dataset"
    meshes_root = os.path.join(root, "DFCData", "meshes")
    generate_object_poses(meshes_root)
    generate_object_pc(meshes_root)
    generate_object_table_pc(meshes_root)
    write_splits(meshes_root)
    print("End")