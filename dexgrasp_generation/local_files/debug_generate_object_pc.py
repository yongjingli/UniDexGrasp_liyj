"""
Last modified date: 2023.06.06
Author: Jialiang Zhang
Description: Generate object point clouds
"""

import os
import time

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import transforms3d
import torch
import pytorch3d.io
import pytorch3d.ops
import pytorch3d.structures
import sapien.core as sapien
from multiprocessing import Pool, current_process
from tqdm import tqdm
from sapien.utils.viewer import Viewer

from utils_data import save_2_ply


def sample_projected(_):
    args, object_code, idx = _

    # worker = current_process()._identity[0]
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list[(worker - 1) % len(args.gpu_list)]
    print(idx)

    object_path = os.path.join(args.data_root_path, object_code, 'coacd', 'decomposed.obj')

    # set simulator

    engine = sapien.Engine()
    # engine.set_log_level('critical')
    # renderer = sapien.VulkanRenderer(offscreen_only=True)
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.add_ground(altitude=0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer.set_camera_xyz(x=-4, y=0, z=2)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)


    # rscene = scene.get_renderer_scene()
    # rscene = scene
    # 设置了环境光的强度，值范围通常在 [0, 1] 之间，影响整个场景的基础亮
    # 一个方向光源，光源方向为 [0, 1, -1]（即从上方照射），颜色为 [0.5, 0.5, 0.5]，并且启用阴影
    # 添加多个点光源
    # rscene.set_ambient_light([0.5, 0.5, 0.5])
    # rscene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    # rscene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    # rscene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    # rscene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    # 创建一些常规形状的obj
    # builder = scene.create_actor_builder()
    # builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
    # builder.add_box_visual(half_size=[0.5, 0.5, 0.5], color=[1., 0., 0.])
    # box = builder.build(name='box')  # Add a box
    # box.set_pose(sapien.Pose(p=[0, 0, 0.5]))



    # 这行代码构建一个运动学演员，意味着该对象不会受到物理引擎的重力或碰撞影响，可以通过代码控制其运动。
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(object_path, scale=[args.scale, args.scale, args.scale])
    object_actor = builder.build_kinematic()

    # 创建一个运动学演员 camera_mount_actor，该演员可以用于安装相机并通过代码控制其移动。
    camera_mount_actor = scene.create_actor_builder().build_kinematic()

    # name="camera"：相机的名称。
    # actor=camera_mount_actor：将相机安装在之前创建的运动学演员上。
    # pose=sapien.Pose()：相机相对于安装演员的位置，默认是原点（即与演员的坐标相同）。
    # width 和 height：相机的分辨率。
    # fovx=0：水平视场角（可选，通常不使用）。
    # fovy=np.deg2rad(35)：垂直视场角，设置为 35 度。
    # near 和 far：相机的近裁剪面和远裁剪面，定义可见范围。
    camera = scene.add_mounted_camera(
        name="camera",
        actor=camera_mount_actor,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=args.width,
        height=args.height,
        fovx=0,
        fovy=np.deg2rad(35),
        near=args.near,
        far=args.far,
    )
    # print('Intrinsic matrix\n', camera.get_camera_matrix())

    # camera_eye = np.array([[0.2, -0.5, 1.0], [1.0, 0.2, 1.0], [0.2, 0.2, 1.4]])
    # camera_forward = np.array([[0, 4.0, 0], [-4.0, 0, 0], [0, -0.01, -2.0]]
    camera_forward = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    camera_eye = np.array([0, 0, args.camera_height]) - args.camera_distance * camera_forward

    # angles = np.linspace(0, 2 * np.pi, args.n_cameras, endpoint=False)
    # cam_pos_array = np.stack([
    #     args.camera_distance * np.sin(args.theta) * np.cos(angles),
    #     args.camera_distance * np.sin(args.theta) * np.sin(angles),
    #     args.camera_distance * np.cos(args.theta).repeat(args.n_cameras)
    # ], axis=1)
    # cam_pos_array = np.stack(np.meshgrid([-args.camera_distance, args.camera_distance], [-args.camera_distance, args.camera_distance], [-args.camera_distance, args.camera_distance]), axis=-1).reshape(-1, 3)
    # print(f'n_camera: {len(cam_pos_array)}')

    # load poses

    pose_matrices = np.load(os.path.join(args.data_root_path, object_code, 'poses.npy'))
    pose_matrices = pose_matrices if len(pose_matrices) <= args.n_poses else pose_matrices[:args.n_poses]

    pcs = []

    if os.path.exists(os.path.join(args.data_root_path, object_code, 'pcs.npy')):
        pcs_old = np.load(os.path.join(args.data_root_path, object_code, 'pcs.npy'))
        for pc in pcs_old:
            pcs.append(pc)
        # if len(pcs) >= args.n_poses:
        #     return

    # for pose_matrix in pose_matrices[len(pcs):]:
    for i, pose_matrix in enumerate(pose_matrices):
        # pose_matrix = pose_matrices[70]
        # print(pose_matrix)
        pc = []

        translation = pose_matrix[:3, 3]
        translation *= args.scale
        translation[:2] = 0
        rotation_matrix = pose_matrix[:3, :3]
        rotation_quaternion = transforms3d.quaternions.mat2quat(rotation_matrix)

        while not viewer.closed:
            # for cam_pos in cam_pos_array:
            # 设置从上面和四面的方向获取物体的点云，如果是底部的点云的话就会缺失
            # camera_forward为设置照射的方向, 将[1, 0, 0]作为新x轴forward的方向, 方向向量的新基
            # camera_forward = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)

            # camera_eye为设置照射的距离, 可以看到xy分别为0，0所以就是为将pose的xy设置为0，作为基准 translation[:2] = 0
            # cam原来的位置在[0 0 0.05]的地方，xy左右移动0.5米，往上移动0.5的位置
            # camera_eye = np.array([0, 0, args.camera_height]) - args.camera_distance * camera_forward

            for idx_camera in range(len(camera_eye)):
                # Compute the camera pose by specifying forward(x), left(y) and up(z)
                cam_pos = camera_eye[idx_camera]
                forward = camera_forward[idx_camera]

                # forward = -cam_pos / np.linalg.norm(cam_pos)
                # forward包括以下的向量 [[ 1.  0.  0.], [-1.  0.  0.], [ 0.  1.  0.], [ 0. -1.  0.], [ 0.  0. -1.]]
                # 转换的目标是forward转换为新的x轴方向， 如[-1.  0.  0.]为将x轴置反

                left = np.cross([0, 0, 1], forward)

                # np.linalg.norm(left) < 0.01 判断是否与z轴接近，若是则修改为与y轴进行叉乘, 避免forward就是z轴
                left = np.cross([0, 1, 0], forward) if np.linalg.norm(left) < 0.01 else left

                # 归一化left方向向量，left与forward是垂直
                left = left / np.linalg.norm(left)

                # 得到up向量，与forward和left都是垂直，这样就是两两垂直的新基
                up = np.cross(forward, left)

                # 构建新基，调转方向,构建旋转矩阵
                mat44 = np.eye(4)
                mat44[:3, :3] = np.stack([forward, left, up], axis=1)
                mat44[:3, 3] = cam_pos

                camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

                # render 设置objec的位置
                object_actor.set_pose(sapien.Pose(translation, rotation_quaternion))
                scene.step()  # make everything set
                scene.update_render()
                camera.take_picture()

                # 得到物体在相机中的位置
                # Each pixel is (x, y, z, is_valid) in camera space (OpenGL/Blender)
                position = camera.get_float_texture('Position')  # [H, W, 4]
                # OpenGL/Blender: y up and -z forward
                points_opengl = position[..., :3][position[..., 2] != 0]
                # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
                # camera.get_model_matrix() must be called after scene.update_render()!
                model_matrix = camera.get_model_matrix()

                # model_matrix为相机到世界坐标的位置，将相机坐标的点转到世界坐标上
                # 这里写成旋转矩阵右乘的形式，等于左乘这个矩阵的逆，相当于model_matrix[:3, :3] @ points_opengl
                points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
                pc.append(points_world)

                time.sleep(1)
                scene.update_render()  # Update the world to the renderer
                viewer.render()

        pc = np.concatenate(pc)
        pc_down_sampled = pc[np.random.choice(len(pc), args.max_n_points, replace=False)] if len(
            pc) > args.max_n_points else pc
        pc_sampled = \
        pytorch3d.ops.sample_farthest_points(torch.tensor(pc_down_sampled).unsqueeze(0), K=args.num_samples)[0][0]

        # 又转回到0 0 0 mesh模型中的坐标中,跟原来的区别可能就是少了一些在相应的pose中相机看不到的点云
        # 这里写成旋转矩阵右乘的形式，等于左乘这个矩阵的逆，这里的旋转矩阵为物体到世界坐标的旋转
        pc_sampled = (pc_sampled - translation) @ rotation_matrix / args.scale
        pcs.append(pc_sampled)
        s_ply_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/object_pcs/object_pc.ply"
        save_2_ply(s_ply_path, pcs[20][:, 0], pcs[20][:, 1], pcs[20][:, 2], color=None)
        print(pcs[0])

        exit(1)

    # pcs = np.stack(pcs)
    # np.save(os.path.join(args.data_root_path, object_code, 'pcs.npy'), pcs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiments settings
    # parser.add_argument('--data_root_path', type=str, default='../data/DFCData/meshes')
    parser.add_argument('--data_root_path', type=str, default='./data/DFCData/meshes')
    parser.add_argument('--n_poses', type=int, default=100)
    parser.add_argument('--max_n_points', type=int, default=9000)
    parser.add_argument('--num_samples', type=int, default=3000)
    parser.add_argument('--n_cpu', type=int, default=8)
    # parser.add_argument('--n_cameras', type=int, default=6)
    # parser.add_argument('--theta', type=float, default=np.pi / 4)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--gpu_list', type=str, nargs='*', default=['0', '1', '2', '3'])
    # camera settings
    parser.add_argument('--camera_distance', type=float, default=0.5)
    parser.add_argument('--camera_height', type=float, default=0.05)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--near', type=float, default=0.1)
    parser.add_argument('--far', type=float, default=100)
    args = parser.parse_args()

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

    # 生成不同scale下的点云
    # 需要对物体进行scale，需要对位姿的translate进行scale
    for param in parameters:
        print("param:", param)
        sample_projected(param)

    # with Pool(args.n_cpu) as p:
    #     it = tqdm(p.imap(sample_projected, parameters), desc='sampling', total=len(parameters))
    #     list(it)
