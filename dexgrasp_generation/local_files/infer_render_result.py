import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.realpath('.'))

import open3d as o3d
import argparse
import numpy as np
import trimesh as tm
import torch
import pytorch3d.transforms
import plotly.graph_objects as go
from datasets.shadow_hand_builder import ShadowHandBuilder

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]

# 渲染 Mesh 函数
def render_mesh(vertices, faces):
    glBegin(GL_TRIANGLES)
    for face in faces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

# 初始化 OpenGL
def init_opengl():
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='eval')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--canonical_frame', type=int, default=0)
    args = parser.parse_args()
    
    # load data
    # result = torch.load(os.path.join(args.exp_dir, 'result.pt'), map_location='cpu')
    
    # hand model
    builder = ShadowHandBuilder()

    result_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/infer_result"
    # result_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/infer_result_seed233"
    # result_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/infer_result_seed1024"
    # result_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/infer_result_seed2048"
    result_names = [name for name in os.listdir(result_root) if name.endswith(".pt")]

    print(result_names)
    for result_name in result_names:
        result = torch.load(os.path.join(result_root, result_name), map_location='cpu')

        # object mesh
        object_code = result['object_code'][args.num]
        object_scale = result['scale'][args.num].item()
        object_mesh = tm.load(os.path.join('data/DFCData/meshes', object_code, 'coacd/decomposed.obj')).apply_scale(object_scale)
        object_pc = result['canon_obj_pc' if args.canonical_frame else 'obj_pc'][args.num].numpy()
        object_pc_plotly = go.Scatter3d(
            x=object_pc[:, 0],
            y=object_pc[:, 1],
            z=object_pc[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='lightgreen',
            )
        )

        # hand mesh
        hand_translation = result['sampled_rotation'][args.num].numpy().T @ result['translation'][args.num].numpy() if args.canonical_frame else result['translation'][args.num].numpy()
        hand_rotation = np.eye(3) if args.canonical_frame else result['sampled_rotation'][args.num].numpy()
        hand_qpos = result['hand_qpos'][args.num].numpy()
        # hand_mesh = builder.get_hand_mesh(
        #     rotation_mat=hand_rotation,
        #     world_translation=hand_translation,  # hand_translation,
        #     qpos=hand_qpos,
        # )

        hand_mesh = builder.get_hand_mesh(
            rotation_mat=torch.eye(3),
            world_translation=torch.tensor([0.0, 0.0, 3.0]),  # hand_translation,
            qpos=torch.zeros_like(hand_qpos),
        )



        hand_mesh = tm.Trimesh(
            vertices=hand_mesh.verts_list()[0].numpy(),
            faces=hand_mesh.faces_list()[0].numpy()
        )
        hand_mesh_plotly = go.Mesh3d(
            x=hand_mesh.vertices[:, 0],
            y=hand_mesh.vertices[:, 1],
            z=hand_mesh.vertices[:, 2],
            i=hand_mesh.faces[:, 0],
            j=hand_mesh.faces[:, 1],
            k=hand_mesh.faces[:, 2],
            color='lightblue',
            opacity=0.5,
        )
        tta_hand_translation = result['sampled_rotation'][args.num].numpy().T @ result['tta_hand_pose'][args.num][:3].numpy() if args.canonical_frame else result['tta_hand_pose'][args.num][:3].numpy()
        tta_hand_rotation = np.eye(3) if args.canonical_frame else pytorch3d.transforms.axis_angle_to_matrix(result['tta_hand_pose'][args.num][3:6]).numpy()
        tta_hand_qpos = result['tta_hand_pose'][args.num][6:].numpy()
        tta_hand_mesh = builder.get_hand_mesh(
            rotation_mat=tta_hand_rotation,
            world_translation=tta_hand_translation,
            qpos=tta_hand_qpos,
        )
        tta_hand_mesh = tm.Trimesh(
            vertices=tta_hand_mesh.verts_list()[0].numpy(),
            faces=tta_hand_mesh.faces_list()[0].numpy()
        )
        tta_hand_mesh_plotly = go.Mesh3d(
            x=tta_hand_mesh.vertices[:, 0],
            y=tta_hand_mesh.vertices[:, 1],
            z=tta_hand_mesh.vertices[:, 2],
            i=tta_hand_mesh.faces[:, 0],
            j=tta_hand_mesh.faces[:, 1],
            k=tta_hand_mesh.faces[:, 2],
            color='lightblue',
            opacity=1,
        )

        np.save("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_data/vertices.npy",tta_hand_mesh.vertices)
        np.save("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_data/faces.npy", tta_hand_mesh.faces)
        # m = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(tta_hand_mesh.vertices),
        #                               o3d.open3d.utility.Vector3iVector(tta_hand_mesh.faces))
        # m.compute_vertex_normals()


        # oepn3d
        # o3d.visualization.draw_geometries([m])

        # plotly
        # visualize
        # fig = go.Figure([object_pc_plotly, hand_mesh_plotly, tta_hand_mesh_plotly])
        # fig = go.Figure([object_pc_plotly, hand_mesh_plotly])
        # fig = go.Figure([object_pc_plotly, tta_hand_mesh_plotly])
        # fig.write_image('p_pie2.png', engine="kaleido")

        # opengl
        # pygame.init()
        # display = (800, 600)
        # pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        # gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        # glTranslatef(0.0, 0.0, -5)
        #
        # init_opengl()
        #
        # # 渲染循环
        # while True:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             # return 0
        #
        #     glRotatef(1, 0, 1, 0)  # 旋转模型
        #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #     render_mesh(tta_hand_mesh.vertices, tta_hand_mesh.faces)
        #
        #     # 捕获并保存图像
        #     glReadBuffer(GL_FRONT)
        #     pixels = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
        #     image = pygame.image.fromstring(pixels, display, "RGB")
        #     pygame.image.save(image, "rendered_image.png")  # 保存图像
        #
        #     pygame.display.flip()
        #     pygame.time.wait(10)
        #
        exit(1)

        # fig.update_layout(scene_aspectmode='data')
        # fig.show()
