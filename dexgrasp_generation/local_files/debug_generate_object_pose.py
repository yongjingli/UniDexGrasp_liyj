"""
Last modified date: 2023.06.06
Author: Jialiang Zhang
Description: Generate object pose, random free-fall, use SAPIEN
"""

import os

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import transforms3d
from multiprocessing import Pool
import sapien.core as sapien
from tqdm import tqdm
import time
from sapien.utils.viewer import Viewer


def generate_object_pose(_):
    args, object_code = _

    if not args.overwrite and os.path.exists(os.path.join(args.data_root_path, object_code, 'poses.npy')):
        return

    # set simulator

    engine = sapien.Engine()
    # engine.set_log_level('error')

    scene = engine.create_scene()
    scene.set_timestep(args.time_step)
    scene_config = sapien.SceneConfig()
    scene_config.default_restitution = args.restitution

    scene.add_ground(altitude=0, render=False)

    # load object

    if os.path.exists(os.path.join(args.data_root_path, object_code, 'coacd', 'coacd.urdf')) and not os.path.exists(
            os.path.join(args.data_root_path, object_code, 'coacd', 'coacd_convex_piece_63.obj')):
        loader = scene.create_urdf_loader()
        loader.fix_root_link = False
        object_actor = loader.load(os.path.join(args.data_root_path, object_code, 'coacd', 'coacd.urdf'))
        object_actor.set_name(name='object')
    else:
        builder = scene.create_actor_builder()
        builder.add_collision_from_file(os.path.join(args.data_root_path, object_code, 'coacd', 'decomposed.obj'))
        object_actor = builder.build(name='object')

    # generate object pose
    pose_matrices = []
    for i in range(args.n_samples):
        # random pose
        translation = np.zeros(3)
        translation[2] = 1 + np.random.rand()
        rotation_euler = 2 * np.pi * np.random.rand(3)
        rotation_quaternion = transforms3d.euler.euler2quat(*rotation_euler, axes='sxyz')
        try:
            object_actor.set_root_pose(sapien.Pose(translation, rotation_quaternion))
        except AttributeError:
            object_actor.set_pose(sapien.Pose(translation, rotation_quaternion))
        # simulate
        for t in range(args.sim_steps):
            scene.step()
        pose_matrices.append(object_actor.get_pose().to_transformation_matrix())
    pose_matrices = np.stack(pose_matrices)

    # save results

    # np.save(os.path.join(args.data_root_path, object_code, 'poses.npy'), pose_matrices)
    #
    # # remove convex hull
    # if os.path.exists(os.path.join(args.data_root_path, object_code, 'coacd', 'decomposed.obj.convex.stl')):
    #     os.remove(os.path.join(args.data_root_path, object_code, 'coacd', 'decomposed.obj.convex.stl'))


def visualize_scence():
    engine = sapien.Engine()
    # engine.set_log_level('error')

    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(args.time_step)
    scene_config = sapien.SceneConfig()
    scene_config.default_restitution = args.restitution

    # scene.add_ground(altitude=0, render=False)
    scene.add_ground(altitude=0)

    decomposed_obj_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/data/DFCData/meshes/core/bottle-1a7ba1f4c892e2da30711cdbdbc73924/coacd/decomposed.obj"
    builder = scene.create_actor_builder()
    # 缺少dae文件，无法可视化
    # builder.add_collision_from_file(os.path.join(args.data_root_path, object_code, 'coacd', 'decomposed.obj'))
    # builder.add_collision_from_file(filename=decomposed_obj_path)
    # builder.add_collision_from_file(filename='../assets/banana/collision_meshes/collision.obj')
    # builder.add_visual_from_file(filename='../assets/banana/visual_meshes/visual.dae')
    # object_actor = builder.build(name='object')
    # object_actor.set_color([1.0, 0.5, 0.2])  # 设置为橙色
    # object_actor.set_root_pose(sapien.Pose(translation, rotation_quaternion))
    # object_actor.set_root_pose(sapien.Pose(p=[0, 0, 0.5]))
    # object_actor.set_pose(sapien.Pose(p=[0, 0, 0.5]))

    # 创建一些常规形状的obj
    # builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
    # builder.add_box_visual(half_size=[0.5, 0.5, 0.5], color=[1., 0., 0.])
    # box = builder.build(name='box')  # Add a box
    # box.set_pose(sapien.Pose(p=[0, 0, 0.5]))


    # Load URDF
    urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/data/DFCData/meshes/core/bottle-1a7ba1f4c892e2da30711cdbdbc73924/coacd/coacd.urdf"
    # loader: sapien.URDFLoader = scene.create_urdf_loader()
    # loader.fix_root_link = True
    # robot: sapien.Articulation = loader.load(urdf_path)
    # robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    # loader.fix_root_link = False           # 扔下的时候会出现一个随机的pose
    object_actor = loader.load(urdf_path)
    object_actor.set_name(name='object')

    # 设置相机
    viewer = Viewer(renderer)  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=-4, y=0, z=2)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    while not viewer.closed:  # Press key q to quit
        scene.step()  # Simulate the world

        # random pose
        translation = np.zeros(3)
        translation[2] = 1 + np.random.rand()
        rotation_euler = 2 * np.pi * np.random.rand(3)
        rotation_quaternion = transforms3d.euler.euler2quat(*rotation_euler, axes='sxyz')
        object_actor.set_root_pose(sapien.Pose(translation, rotation_quaternion))

        print(object_actor.get_pose().to_transformation_matrix())
        time.sleep(1)

        scene.update_render()  # Update the world to the renderer
        viewer.render()


def check_poses():
    pose_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/data/DFCData/meshes/core/bottle-1a7ba1f4c892e2da30711cdbdbc73924/poses.npy"
    poses = np.load(pose_path)
    # print(poses[:, 2, ])
    print(np.max(poses[:, :3, 3]))
    print(np.min(poses[:, :3, 3]))
    print(poses[0, :3, 3])
    print(poses[0])

    print(np.max(poses[:, 0, 3]))
    print(np.min(poses[:, 0, 3]))

    print(np.max(poses[:, 1, 3]))
    print(np.min(poses[:, 1, 3]))

    print(np.max(poses[:, 2, 3]))
    print(np.min(poses[:, 2, 3]))

    # 实际上poses里的平移在xyz方向上都有，并不是生成代码里只是生成了z方向上, 数值的范围应该是[-1, 1]
    # 但是在实际的应用，进行pc或者pc_table 或者dex_dataset加载的时候都进行了让xy偏移设置为0的操作
    # pose_matrix[:2, 3] = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    # parser.add_argument('--data_root_path', type=str, default='../data/DFCData/meshes')
    parser.add_argument('--data_root_path', type=str, default='./data/DFCData/meshes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--overwrite', type=bool, default=True)
    parser.add_argument('--n_cpu', type=int, default=16)
    # simulator settings
    parser.add_argument('--sim_steps', type=int, default=1000)
    parser.add_argument('--time_step', type=float, default=1 / 100)
    parser.add_argument('--restitution', type=float, default=0.01)

    args = parser.parse_args()

    # seed
    np.random.seed(args.seed)

    # load object list
    object_code_list = os.listdir(args.data_root_path)

    # generate object pose
    # for object_code in tqdm(object_code_list, desc='generating'):
    #     generate_object_pose((args, object_code))

    # visualize poses generate
    # visualize_scence()

    # check pose
    check_poses()

    # with Pool(args.n_cpu) as p:
    #     param_list = []
    #     for object_code in object_code_list:
    #         param_list.append((args, object_code))
    #     list(tqdm(p.imap(generate_object_pose, param_list), desc='generating', total=len(param_list), miniters=1))
