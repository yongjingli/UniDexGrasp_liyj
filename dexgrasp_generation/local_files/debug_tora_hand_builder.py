import copy

import pytorch_kinematics as pk
from pytorch3d.structures import Meshes, join_meshes_as_batch
import torch
import os

import numpy as np
import trimesh
import trimesh as tm
import open3d as o3d
from utils_data import save_2_ply


class ToraHandBuilder():
    joint_names = ['wmzcb', 'wmzjdxz', 'wmzzdxz', 'wmzydxz', 'zzcb', 'zzjdxz', 'zzzdxz', 'zzydxz', 'szcb', 'szjdxz', 'szzdxz', 'szydxz', 'mzcb', 'mzjdxz', 'mzzd', 'mzydxz']

    mesh_filenames = [  "forearm_electric.obj",
                        "forearm_electric_cvx.obj",
                        "wrist.obj",
                        "palm.obj",
                        "knuckle.obj",
                        "F3.obj",
                        "F2.obj",
                        "F1.obj",
                        "lfmetacarpal.obj",
                        "TH3_z.obj",
                        "TH2_z.obj",
                        "TH1_z.obj"]

    def __init__(self,
                 mesh_dir="data/mjcf/meshes",
                 urdf_path="data/mjcf/shadow_hand.xml",
                 kpt_infos=None):
        # self.chain = pk.build_chain_from_mjcf(open(urdf_path).read()).to(dtype=torch.float)
        self.chain = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read())

        self.mesh = {}
        self.key_pts = []
        device = 'cpu'

        def build_mesh_recurse(body):
            if(len(body.link.visuals) > 0):
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
                    if visual.geom_type == "box":
                        link_mesh = trimesh.primitives.Box(extents=2 * visual.geom_param)
                    elif visual.geom_type == "capsule":
                        link_mesh = trimesh.primitives.Capsule(radius=visual.geom_param[0], height=visual.geom_param[1]*2).apply_translation((0, 0, -visual.geom_param[1]))
                    elif visual.geom_type == "mesh":
                        # link_mesh = trimesh.load_mesh(os.path.join(mesh_dir, visual.geom_param[0].split(":")[1]+".obj"), process=False)
                        link_mesh = trimesh.load_mesh(os.path.join(mesh_dir, visual.geom_param.split("/")[-1]), process=False)
                        # if visual.geom_param[1] is not None:
                        #     scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)
                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                    pos = visual.offset.to(device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                self.mesh[link_name] = {'vertices': link_vertices,
                                        'faces': link_faces,
                                        }
            for children in body.children:
                build_mesh_recurse(children)

        def build_keys(kpt_infos):
            for pt_info in kpt_infos:
                _, link_name, vect_ind = pt_info
                pt = self.mesh[link_name]["vertices"][vect_ind]
                self.key_pts.append([link_name, torch.unsqueeze(pt, dim=0)])

        build_mesh_recurse(self.chain._root)
        if kpt_infos is not None:
            build_keys(kpt_infos)

    def qpos_to_qpos_dict(self, qpos,
                          hand_qpos_names=None):
        """
        :param qpos: [24]
        WARNING: The order must correspond with the joint_names
        """
        if hand_qpos_names is None:
            hand_qpos_names = ToraHandBuilder.joint_names
        assert len(qpos) == len(hand_qpos_names)
        return dict(zip(hand_qpos_names, qpos))

    def qpos_dict_to_qpos(self, qpos_dict,
                          hand_qpos_names=None):
        """
        :return: qpos: [24]
        WARNING: The order must correspond with the joint_names
        """
        if hand_qpos_names is None:
            hand_qpos_names = ToraHandBuilder.joint_names
        return np.array([qpos_dict[name] for name in hand_qpos_names])

    def get_hand_mesh(self,
                      rotation_mat,
                      world_translation,
                      qpos=None,
                      hand_qpos_dict=None,
                      hand_qpos_names=None,
                      without_arm=False):
        """
        Either qpos or qpos_dict should be provided.
        :param qpos: [24] numpy array
        :rotation_mat: [3, 3]
        :world_translation: [3]
        :return:
        """
        if qpos is None:
            if hand_qpos_names is None:
                hand_qpos_names = ToraHandBuilder.joint_names
            assert hand_qpos_dict is not None, "Both qpos and qpos_dict are None!"
            qpos = np.array([hand_qpos_dict[name] for name in hand_qpos_names], dtype=np.float32)
        current_status = self.chain.forward_kinematics(qpos[np.newaxis, :])

        meshes = []

        for link_name in self.mesh:
            v = current_status[link_name].transform_points(self.mesh[link_name]['vertices'])
            v = v @ rotation_mat.T + world_translation
            f = self.mesh[link_name]['faces']
            meshes.append(Meshes(verts=[v], faces=[f]))

        if without_arm:
            meshes = join_meshes_as_batch(meshes[1:])  # each link is a "batch"
        else:
            meshes = join_meshes_as_batch(meshes)  # each link is a "batch"
        return Meshes(verts=[meshes.verts_packed().type(torch.float32)],
                      faces=[meshes.faces_packed()])

    def get_hand_points(self,
                      rotation_mat,
                      world_translation,
                      qpos=None,
                      hand_qpos_dict=None,
                      hand_qpos_names=None,
                      without_arm=False):
        """
        Either qpos or qpos_dict should be provided.
        :param qpos: [24] numpy array
        :rotation_mat: [3, 3]
        :world_translation: [3]
        :return:
        """
        if qpos is None:
            if hand_qpos_names is None:
                hand_qpos_names = ToraHandBuilder.joint_names
            assert hand_qpos_dict is not None, "Both qpos and qpos_dict are None!"
            qpos = np.array([hand_qpos_dict[name] for name in hand_qpos_names], dtype=np.float32)
        current_status = self.chain.forward_kinematics(qpos[np.newaxis, :])

        points = []

        # self.key_pts
        for key_pt in self.key_pts:
            link_name, pt = key_pt
            v = current_status[link_name].transform_points(pt)
            v = v @ rotation_mat.T + world_translation
            points.append(v)
        points = torch.concat(points)
        return points


def debug_tora_hand():
    mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_R/meshes"
    urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_R/urdf/ZY_R.urdf"
    tora_hand_builder = ToraHandBuilder(mesh_dir, urdf_path)

    print(len(tora_hand_builder.mesh))

    # ['wmzcb', 'wmzjdxz', 'wmzzdxz', 'wmzydxz', 'zzcb', 'zzjdxz', 'zzzdxz', 'zzydxz', 'szcb', 'szjdxz', 'szzdxz', 'szydxz', 'mzcb', 'mzjdxz', 'mzzd', 'mzydxz'
    print(tora_hand_builder.chain.get_joint_parameter_names())

    hand_rotation = torch.eye(3)
    hand_translation = torch.zeros(3)
    hand_qpos = torch.zeros(16)

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

    hand_mesh = tora_hand_builder.get_hand_mesh(
        rotation_mat=hand_rotation,
        world_translation=hand_translation,
        qpos=hand_qpos,
    )
    hand_mesh = tm.Trimesh(
        vertices=hand_mesh.verts_list()[0].numpy(),
        faces=hand_mesh.faces_list()[0].numpy()
    )

    s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample"
    # hand_mesh.export(os.path.join(s_root, "tora_hand.obj"))

    print("fffff")


def debug_get_joint_key_point():
    # mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_R/meshes"
    # urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_R/urdf/ZY_R.urdf"

    # R_hand_pt_infos = [[0, 'yz', 94163], [0, 'yz', 183389], [0, 'yz', 187314], [0, 'yz', 191257], [0, 'yz', 93356],
    #                    # 手掌位置
    #                    [2, 'wmzjd', 122], [6, 'zzjd', 122], [10, 'szjd', 122],  # 实指,中指向,无名指的第一个关节
    #                    [3, 'wmzzd', 4246], [7, 'zzzd', 4246], [11, 'szzd', 4246],  # 实指,中指向,无名指的第二个关节
    #                    [4, 'wmzyd', 98867], [8, 'zzyd', 98867], [12, 'szyd', 98867],  # 实指,中指向,无名指的第三个关节
    #                    [15, 'mzzd', 3386], [15, 'mzzd', 3431],  # 拇指的第一个和第二个关节
    #                    [16, 'mzyd', 87084],  # 拇指的第三个关节
    #                    ]

    mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/meshes"
    urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/urdf/ZY_L.urdf"

    L_hand_pt_infos = [[0, 'zsz', 90047], [0, 'zsz', 180340], [0, 'zsz', 184293], [0, 'zsz', 188224], [0, 'zsz', 19004],
                       # 手掌位置
                       [2, 'wmzjd', 122], [6, 'zzjd', 122], [10, 'szjd', 122],  # 实指,中指向,无名指的第一个关节
                       [3, 'wmzzd', 4246], [7, 'zzzd', 4246], [11, 'szzd', 4246],  # 实指,中指向,无名指的第二个关节
                       [4, 'wmzyd', 98867], [8, 'zzyd', 98867], [12, 'szyd', 98867],  # 实指,中指向,无名指的第三个关节
                       [15, 'mzzd', 3386], [15, 'mzzd', 3431],  # 拇指的第一个和第二个关节
                       [16, 'mzyd', 87084],  # 拇指的第三个关节
                       ]

    # mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/meshes"
    # urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/urdf/ZY_L.urdf"

    tora_hand_builder = ToraHandBuilder(mesh_dir, urdf_path, kpt_infos=L_hand_pt_infos)
    hand_rotation = torch.eye(3)
    hand_translation = torch.zeros(3)
    hand_qpos = torch.zeros(16)

    hand_mesh = tora_hand_builder.get_hand_mesh(
        rotation_mat=hand_rotation,
        world_translation=hand_translation,
        qpos=hand_qpos,
    )
    hand_mesh = tm.Trimesh(
        vertices=hand_mesh.verts_list()[0].numpy(),
        faces=hand_mesh.faces_list()[0].numpy()
    )

    hand_mesh.export("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/tora_point.obj")

    hand_points = tora_hand_builder.get_hand_points(
        rotation_mat=hand_rotation,
        world_translation=hand_translation,
        qpos=hand_qpos,
    )
    hand_points = hand_points.detach().cpu().numpy()
    save_2_ply("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/pts.ply", hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], color=None)


def find_point_index_numpy(point_cloud, target_point):
    points = np.array(point_cloud)
    # 计算每个点与目标点的差值
    diff = points - target_point
    # 计算差值的平方和
    squared_distances = np.sum(diff * diff, axis=1)
    # 找到距离最小的点的索引（假设目标点唯一存在）
    min_index = np.argmin(squared_distances)
    # 检查最小距离是否在一定阈值内，以确定是否真正找到目标点
    if np.sqrt(squared_distances[min_index]) < 1e-6:
        return min_index
    return -1


def get_kpt_debug_data():
    # 验证多次forward_kinematics, 每次都是在原始状态下进行的计算
    if 0:
        # 从URDF文件创建运动链
        from pytorch_kinematics.chain import Chain
        chain = pk.build_chain_from_urdf(open(urdf_path, mode="rb").read())

        # 创建随机的关节角度（这里假设关节是旋转关节，你可以根据实际情况调整）
        batch_size = 1
        joint_names = chain.get_joint_parameter_names()
        joint_angles = torch.rand(batch_size, len(joint_names))

        # 正向运动学计算
        # 齐次变换矩阵表示从运动链的基础（base）坐标系到每个连杆坐标系的变换关系(都是相对于base坐标系的坐标)
        # forward 一次不会改变原来的状态，两次的结果一样
        fk = chain.forward_kinematics(joint_angles)
        print(fk)
        fk = chain.forward_kinematics(joint_angles)
        print(fk)

    if 0:
        # 保存每个linkmesh的点云
        # mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_R/meshes"
        # urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_R/urdf/ZY_R.urdf"

        mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/meshes"
        urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/urdf/ZY_L.urdf"

        tora_hand_builder = ToraHandBuilder(mesh_dir, urdf_path, kpt_infos=None)
        rotation_mat = torch.eye(3)
        world_translation = torch.zeros(3)
        qpos = torch.zeros(16)

        current_status = tora_hand_builder.chain.forward_kinematics(qpos[np.newaxis, :])

        # s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/r_hand"
        s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/l_hand"

        for i, link_name in enumerate(tora_hand_builder.mesh):
            v = current_status[link_name].transform_points(tora_hand_builder.mesh[link_name]['vertices'])
            v = v @ rotation_mat.T + world_translation
            s_ply_path = os.path.join(s_root, str(i) + "_" + link_name + ".ply")
            save_2_ply(s_ply_path, v[:, 0], v[:, 1], v[:, 2], color=None)
            print(v.shape)


    if 0:
        # 右手的关键点配置
        # R_hand_pt_infos = [[0, 'yz', 94163], [0, 'yz', 183389], [0, 'yz', 187314], [0, 'yz', 191257], [0, 'yz', 93356],
        #                    # 手掌位置
        #                    [2, 'wmzjd', 122], [6, 'zzjd', 122], [10, 'szjd', 122],  # 实指,中指向,无名指的第一个关节
        #                    [3, 'wmzzd', 4246], [7, 'zzzd', 4246], [11, 'szzd', 4246],  # 实指,中指向,无名指的第二个关节
        #                    [4, 'wmzyd', 98867], [8, 'zzyd', 98867], [12, 'szyd', 98867],  # 实指,中指向,无名指的第三个关节
        #                    [15, 'mzzd', 3386], [15, 'mzzd', 3431],  # 拇指的第一个和第二个关节
        #                    [16, 'mzyd', 87084],  # 拇指的第三个关节
        #                    ]

        # 左手的关键点配置
        L_hand_pt_infos = [[0, 'zsz', 90047], [0, 'zsz', 180340], [0, 'zsz', 187314], [0, 'zsz', 191257], [0, 'zsz', 93356],
                           # 手掌位置
                           [2, 'wmzjd', 122], [6, 'zzjd', 122], [10, 'szjd', 122],  # 实指,中指向,无名指的第一个关节
                           [3, 'wmzzd', 4246], [7, 'zzzd', 4246], [11, 'szzd', 4246],  # 实指,中指向,无名指的第二个关节
                           [4, 'wmzyd', 98867], [8, 'zzyd', 98867], [12, 'szyd', 98867],  # 实指,中指向,无名指的第三个关节
                           [15, 'mzzd', 3386], [15, 'mzzd', 3431],  # 拇指的第一个和第二个关节
                           [16, 'mzyd', 87084],  # 拇指的第三个关节
                           ]

        # 对每个link mesh选择关键点
        # 0
        # ply_point = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/0_zsz.ply"
        # [94163, 183389, 187314, 191257, 93356]  # 右手
        # target_points = np.array([[-0.031395, -0.000946, 0.000929],
        #                           [-0.005050, 0.032041, 0.116159],
        #                           [-0.005250, 0.000311, 0.115841],
        #                           [-0.005250, -0.031500, 0.115800],
        #                           [-0.027817, -0.021772, 0.029716],])


        # [90047, 180340, 184293, 188224, 19004]  # 左手
        # target_points = np.array([[-0.031307, 0.003915, 0.001929],
        #                           [-0.005216, -0.031529, 0.11680],
        #                           [-0.005216, 0.000282, 0.116841],
        #                           [-0.005216, 0.031471, 0.116800],
        #                           [-0.029532, 0.023080, 0.026956],])

        # 2 或者 6 或者 10 右手
        # ply_point = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/2.ply"
        # # [122]
        # target_points = np.array([ [-0.00425, 0.0315, 0.153537],])

        # 2 或者 6 或者 10 左手
        ply_point = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/2_wmzjd.ply"
        # [122]
        target_points = np.array([[-0.004216, -0.031529, 0.154538],])

        # 3 或者 7 或者 11
        # ply_point = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/3.ply"
        # # [4246]
        # target_points = np.array([[-0.00425, 0.0315, 0.18217]])

        # 4 或者 8 或者 12
        # ply_point = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/4.ply"
        # # [98867]
        # target_points = np.array([[0.014485, 0.032532, 0.214512]])

        # 15
        # ply_point = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/15.ply"
        # [3386, 3431]
        # target_points = np.array([[-0.009640, -0.087161, 0.046444],
        #                           [-0.009191, -0.130260, 0.034900]])

        # 16
        # ply_point = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/16.ply"
        # # 87084
        # target_points = np.array([[-0.008750, -0.168956, 0.000848]])

        point_cloud = o3d.io.read_point_cloud(ply_point)
        point_cloud = np.array(point_cloud.points)

        target_ind = []
        for target_point in target_points:
            min_ind = find_point_index_numpy(point_cloud, target_point)
            target_ind.append(min_ind)

        print(target_ind)
        target_point = point_cloud[target_ind]

        s_ply_point = ply_point.replace(".ply", "_represtation.ply")
        save_2_ply(s_ply_point, target_point[:, 0],  target_point[:, 1],  target_point[:, 2], color=None)


    if 1:
        # 显示基坐标的位置与坐标系定义
        # s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/r_hand"
        s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tmp/l_hand"

        pcds = []

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)

        names = [name for name in os.listdir(s_root) if name[-4:] in [".ply", ".pcd"]]
        for name in names:
            pcd_path = os.path.join(s_root, name)
            pcd = o3d.io.read_point_cloud(pcd_path)
            colors = np.array([[200/255, 200/255, 200/255] for _ in range(len(pcd.points))])
            pcd.colors = o3d.utility.Vector3dVector(colors)

            pcds.append(pcd)

        pcds.append(axis)
        o3d.visualization.draw_geometries(pcds)


if __name__ == "__main__":
    print("Start")
    # debug_tora_hand()
    # debug_get_joint_key_point()
    get_kpt_debug_data()
    print("End")

