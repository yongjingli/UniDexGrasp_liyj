import os
import sys
import time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.realpath('.'))
import cv2
import torch
from datasets.tora_hand_builder import ToraHandVisualizer
import trimesh as tm
import open3d as o3d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil


def get_hand_params():
    # hand_rotation = torch.eye(3)
    hand_translation = torch.zeros(3)
    # hand_qpos = torch.zeros(16)

    # 调整旋转角度
    roll, pitch, yaw = -90, 0, 0  # roll 对应x轴 pitch y轴
    rot = np.array([roll, pitch, yaw])
    roll, pitch, yaw = np.deg2rad(rot)
    R = o3d.geometry.get_rotation_matrix_from_xyz([roll, pitch, yaw])
    R = R.astype(np.float32)
    hand_rotation = torch.from_numpy(R)

    # 调整平移
    hand_translation[0] = 0.05
    hand_translation[1] = -0.12
    hand_translation[2] = -0.01

    hand_qpos_np = np.zeros(16)

    # hand_qpos_np[12] = 90  # 拇指
    hand_qpos_np[13] = 90  # 拇指
    hand_qpos_np[14] = 40  # 拇指
    hand_qpos_np[15] = 90  # 拇指

    # 无名指
    hand_qpos_np[0] = 0   # 左右
    hand_qpos_np[1] = 10  # 里面那节
    hand_qpos_np[2] = 20
    hand_qpos_np[3] = 50

    # 中指
    hand_qpos_np[4] = 0   # 左右
    hand_qpos_np[5] = 10  # 里面那节
    hand_qpos_np[6] = 20
    hand_qpos_np[7] = 50

    # 食指
    hand_qpos_np[8] = 0   # 左右
    hand_qpos_np[9] = 10  # 里面那节
    hand_qpos_np[10] = 20
    hand_qpos_np[11] = 50
    hand_qpos_np = np.deg2rad(hand_qpos_np).astype(np.float32)
    hand_qpos = torch.from_numpy(hand_qpos_np)

    return hand_rotation, hand_translation, hand_qpos


def debug_tora_hand_visualizer():
    mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_R/meshes"
    urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_R/urdf/ZY_R.urdf"
    tora_hand_visualizer = ToraHandVisualizer(mesh_dir, urdf_path, hand_name="right")

    # mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/meshes"
    # urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/urdf/ZY_L.urdf"
    # tora_hand_visualizer = ToraHandVisualizer(mesh_dir, urdf_path, hand_name="left")

    root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/realsense_09261855"
    img_root = os.path.join(root, "colors")
    vis_img_root = os.path.join(root, "colors_vis")
    pose_root = os.path.join(root, "poses")

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    cam_k_path = os.path.join(root, "cam_k.txt")
    cam_k = np.loadtxt(cam_k_path)
    cam_k = torch.from_numpy(cam_k)

    vis_pose_root = os.path.join(root, "vis_pose")
    if os.path.exists(vis_pose_root):
        shutil.rmtree(vis_pose_root)
    os.mkdir(vis_pose_root)


    for img_name in img_names:

        img_path = os.path.join(img_root, img_name)
        vis_img_path = os.path.join(vis_img_root, img_name)
        pose_path = os.path.join(pose_root, img_name.replace("_color.jpg", "_pose.npy"))

        img = cv2.imread(img_path)
        vis_img = cv2.imread(vis_img_path)
        obj_pose = np.load(pose_path)

        obj_pose = torch.from_numpy(obj_pose)
        hand_rot, hand_trans, hand_qpos = get_hand_params()

        # render_img = tora_hand_visualizer.render_image(img, obj_pose, hand_rot, hand_trans, hand_qpos, cam_k)

        t1 = time.time()
        vis_img = tora_hand_visualizer.vis_grasp_pose(img, obj_pose, hand_rot, hand_trans, hand_qpos, cam_k)
        t2 = time.time() - t1
        print("t2:", t2)

        s_vis_img_path = os.path.join(vis_pose_root, img_name)
        cv2.imwrite(s_vis_img_path, vis_img)

        # plt.imshow(render_img[:, :, ::-1])
        # plt.show()
        # exit(1)


if __name__ == "__main__":
    print("STart")
    debug_tora_hand_visualizer()
    print("End")
