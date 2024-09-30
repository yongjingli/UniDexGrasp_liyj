import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.realpath('.'))

import torch
from datasets.shadow_hand_builder import ShadowHandBuilder
import trimesh as tm
import open3d as o3d
import numpy as np
from PIL import Image


def get_shadow_hand_builder_mesh():
    builder = ShadowHandBuilder()

    hand_rotation = torch.eye(3)
    hand_translation = torch.zeros(3)
    hand_qpos = torch.zeros(22)

    # 调整旋转角度
    roll, pitch, yaw = -90, 0, -90  # roll 对应x轴 pitch y轴
    rot = np.array([roll, pitch, yaw])
    roll, pitch, yaw = np.deg2rad(rot)
    R = o3d.geometry.get_rotation_matrix_from_xyz([roll, pitch, yaw])
    R = R.astype(np.float32)
    hand_rotation = torch.from_numpy(R)

    # 调整平移
    hand_translation[0] = 0.05
    hand_translation[1] = -0.01
    hand_translation[2] = -0.01

    # 调节关节角度
    hand_qpos_np = np.zeros(22)
    # hand_qpos_np[17] = 90   # 17这个关节目前没看到旋转

    hand_qpos_np[19] = 90  # 拇指
    # hand_qpos_np[20] = 90  # 拇指
    # hand_qpos_np[21] = 90  # 拇指

    # 实指
    # hand_qpos_np[0] = 90  # 往外趴
    hand_qpos_np[1] = 90
    hand_qpos_np[2] = 30
    hand_qpos_np[3] = 20

    # 中指
    # hand_qpos_np[4] = 90  # 往外趴
    hand_qpos_np[5] = 90  # 里面那节
    hand_qpos_np[6] = 30
    hand_qpos_np[7] = 20

    # 无名指
    # hand_qpos_np[8] = 90  # 往外趴
    hand_qpos_np[9] = 90
    hand_qpos_np[10] = 30
    hand_qpos_np[11] = 20

    # 尾指
    # hand_qpos_np[12] = 90  # 往外趴
    # hand_qpos_np[13] = 90  # 往内趴
    hand_qpos_np[14] = 90
    hand_qpos_np[15] = 30
    hand_qpos_np[16] = 20

    hand_qpos_np = np.deg2rad(hand_qpos_np).astype(np.float32)
    hand_qpos = torch.from_numpy(hand_qpos_np)

    hand_mesh = builder.get_hand_mesh(
        rotation_mat=hand_rotation,
        world_translation=hand_translation,
        qpos=hand_qpos,
    )
    hand_mesh = tm.Trimesh(
        vertices=hand_mesh.verts_list()[0].numpy(),
        faces=hand_mesh.faces_list()[0].numpy()
    )

    # textures

    # width, height = 512, 512
    # color = (255, 0, 0)  # 红色，你可以根据需要自定义颜色
    # texture_image = Image.new('RGB', (width, height), color)
    # hand_mesh.visual.texture = texture_image

    # s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/shadow_hand_mesh"
    s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample"
    hand_mesh.export(os.path.join(s_root, "shadow_hand.obj"))


def debug_shadow_hand_builder():
    # builder = ShadowHandBuilder()

    import pytorch_kinematics as pk
    mjcf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/data/mjcf/shadow_hand.xml"
    chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float)

    # print(chain)
    # 这里打印的是关节的名称
    print(len(chain.get_joint_parameter_names()), chain.get_joint_parameter_names())
    # ['robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0', 'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1',
    #  'robot0:MFJ0', 'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0', 'robot0:LFJ4', 'robot0:LFJ3',
    #  'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0', 'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1',
    #  'robot0:THJ0']

    # 对关节进行控制
    th = {'robot0:FFJ3': 1, 'robot0:FFJ2': 2.0}
    ret = chain.forward_kinematics(th)

    # 得到每个link_name的位姿
    # {'robot0:palm': Transform3d(rot=tensor([[1., 0., 0., 0.]]), pos=tensor([[0., 0., 0.]])),
    #  'robot0:ffknuckle': Transform3d(rot=tensor([[1., 0., 0., 0.]]), pos=tensor([[0.0330, 0.0000, 0.0950]])),
    print(len(ret.keys()), ret)

    # shadow_hand_builder中采用递归的方式得到所有link_name的mesh
    # build_mesh_recurse
    # 所有的link就是这样一步步递归组装到一起的

    # URDF 对于使用过ROS的读者来说应该并不陌生，它是一种描述机器人模型的协议。
    # MJCF则是MuJoCo为了描述复杂动态系统而设计的一种介于描述语言和程序语言之间的描述协议

    # 对于urdf模型
    # chain = pk.build_chain_from_urdf(open(urdf, mode="rb").read())
    # visualize the frames (the string is also returned)
    # chain.print_tree()
    # print(chain.get_joint_parameter_names())

    print("ffff")


if __name__ == "__main__":
    print("Start")
    # px 读取方式 mujoco
    # pip install mujoco
    # python -m mujoco.viewer

    # 调整hand builder模型
    # get_shadow_hand_builder_mesh()

    # understand
    debug_shadow_hand_builder()

    print("End")