import pytorch_kinematics as pk
from pytorch3d.structures import Meshes, join_meshes_as_batch
import torch
import os
import numpy as np
import trimesh
import trimesh as tm
import cv2
import open3d as o3d


from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    PerspectiveCameras
)


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

        build_mesh_recurse(self.chain._root)

        self.kpt_infos = kpt_infos
        if kpt_infos is not None:
            self.build_key_pts(kpt_infos)

    def build_key_pts(self, kpt_infos):
        for pt_info in kpt_infos:
            _, link_name, vect_ind = pt_info
            pt = self.mesh[link_name]["vertices"][vect_ind]
            self.key_pts.append([link_name, torch.unsqueeze(pt, dim=0)])

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


class ToraHandVisualizer(object):
    def __init__(self, mesh_dir, urdf_path, hand_name="left"):
        assert hand_name in ["left", "right"], "hand_name should be left or right"
        if hand_name == "right":
            key_pt_infos = [[0, 'yz', 94163], [0, 'yz', 183389], [0, 'yz', 187314], [0, 'yz', 191257], [0, 'yz', 93356],
                            [2, 'wmzjd', 122], [6, 'zzjd', 122], [10, 'szjd', 122],
                            [3, 'wmzzd', 4246], [7, 'zzzd', 4246], [11, 'szzd', 4246],
                            [4, 'wmzyd', 98867], [8, 'zzyd', 98867], [12, 'szyd', 98867],
                            [15, 'mzzd', 3386], [15, 'mzzd', 3431],
                            [16, 'mzyd', 87084],
                           ]
        else:
            key_pt_infos = [[0, 'zsz', 90047], [0, 'zsz', 180340], [0, 'zsz', 184293], [0, 'zsz', 188224], [0, 'zsz', 19004],
                            [2, 'wmzjd', 122], [6, 'zzjd', 122], [10, 'szjd', 122],
                            [3, 'wmzzd', 4246], [7, 'zzzd', 4246], [11, 'szzd', 4246],
                            [4, 'wmzyd', 98867], [8, 'zzyd', 98867], [12, 'szyd', 98867],
                            [15, 'mzzd', 3386], [15, 'mzzd', 3431],
                            [16, 'mzyd', 87084],
                           ]

        self.tora_hand_builder = ToraHandBuilder(mesh_dir, urdf_path, kpt_infos=key_pt_infos)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        self.device = device
        self.lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
        self.materials = Materials(device=device,
                                   specular_color=[[1.0, 1.0, 1.0]], # [[0.0, 1.0, 0.0]]
                                   shininess=10.0)

    def get_hand_mesh(self, hand_rot, hand_trans, hand_qpos, color=[200/255, 200/255, 200/255]):
        hand_mesh = self.tora_hand_builder.get_hand_mesh(
            rotation_mat=hand_rot,
            world_translation=hand_trans,
            qpos=hand_qpos,
        )

        verts = hand_mesh.verts_list()[0]
        faces = hand_mesh.faces_list()[0]

        # -Initialize each vertex to bewhite in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        verts_rgb[:, :, 0] = color[0]
        verts_rgb[:, :, 1] = color[1]
        verts_rgb[:, :, 2] = color[2]
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

        mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures
        )
        return mesh

    def render_image(self, img, obj_pose, hand_rot, hand_trans, hand_qpos, cam_k):
        img_h, img_w, _ = img.shape

        R = obj_pose[:3, :3]
        T = obj_pose[:3, 3]

        cameras = self.getcamera(img_w, img_h, R, T, cam_k)
        raster_settings = RasterizationSettings(
            image_size=[img_h, img_w],
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=self.lights
            )
        )

        hand_mesh = self.get_hand_mesh(hand_rot, hand_trans, hand_qpos)
        images = renderer(hand_mesh, lights=self.lights, materials=self.materials, cameras=cameras)
        rendered_image = images[0, ..., :3].cpu().numpy()
        rendered_image = rendered_image[:, :, ::-1]

        fuse_image = img.copy()
        mask = np.all(rendered_image == 1, axis=-1)

        fuse_image[~mask] = fuse_image[~mask] * 0.1 + rendered_image[~mask] * 255 * 0.9
        return fuse_image

    def getcamera(self, width, height, R, T, K):
        T = T.reshape(3)
        R[0, :] = -R[0, :]
        T[0] = -T[0]
        R[1, :] = -R[1, :]
        T[1] = -T[1]
        R = R.t()
        fx, _, cx, _, fy, cy, _, _, _ = K.reshape(9)
        cameras = PerspectiveCameras(
            image_size=[[height, width]],
            R=R[None],
            T=T[None],
            focal_length=torch.tensor([[fx, fy]], dtype=torch.float32),
            principal_point=torch.tensor([[cx, cy]], dtype=torch.float32),
            in_ndc=False,
            device=self.device
        )
        return cameras

    def draw_hand_pose(self, img, uvz):
        uvz = uvz.astype(np.int32)
        line_inds = [[0, 1], [0, 2], [0, 3], [0, 4],
                     [1, 5], [5, 8], [8, 11],
                     [2, 6], [6, 9], [9, 12],
                     [3, 7], [7, 10], [10, 13],
                     [4, 14], [14, 15], [15, 16],
                     ]

        colors = np.zeros((17, 3), dtype=np.uint8)
        colors[0] = [255, 215, 0]
        colors[[1, 5, 8, 11]] = [255, 0, 255]
        colors[[2, 6, 9, 12]] = [0, 255, 0]
        colors[[3, 7, 10, 13]] = [0, 0, 255]
        colors[[4, 14, 15, 16]] = [255, 0, 0]

        for i, _uvz in enumerate(uvz):
            u, v, z = _uvz
            color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
            cv2.circle(img, (u, v), int(5), color, -1)

        for line_ind in line_inds:
            s_i, e_i = line_ind

            pt1 = (uvz[s_i][0], uvz[s_i][1])
            pt2 = (uvz[e_i][0], uvz[e_i][1])
            color = (int(colors[e_i][0]), int(colors[e_i][1]), int(colors[e_i][2]))
            thickness = 2
            cv2.line(img, pt1, pt2, color, thickness)

        return img

    def vis_grasp_pose(self, img, obj_pose, hand_rot, hand_trans, hand_qpos, cam_k):
        hand_points = self.tora_hand_builder.get_hand_points(
            rotation_mat=hand_rot,
            world_translation=hand_trans,
            qpos=hand_qpos,
        )

        obj_pose = obj_pose.float()
        R = obj_pose[:3, :3]
        T = obj_pose[:3, 3]

        hand_points_cam = hand_points @ R.T + T

        cam_fx = cam_k[0, 0]
        cam_fy = cam_k[1, 1]
        cam_cx = cam_k[0, 2]
        cam_cy = cam_k[1, 2]

        x, y, z = hand_points_cam[:, 0], hand_points_cam[:, 1], hand_points_cam[:, 2]

        u = torch.round(x * cam_fx / z + cam_cx)
        v = torch.round(y * cam_fy / z + cam_cy)

        uvz = torch.stack([u, v, z])
        uvz = torch.transpose(uvz, 1, 0).detach().cpu().numpy()

        img = self.draw_hand_pose(img, uvz)
        return img


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


if __name__ == "__main__":
    print("STart")

    mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/meshes"
    urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/urdf/ZY_L.urdf"

    tora_hand_visualizer = ToraHandVisualizer(mesh_dir, urdf_path, hand_name="left")
    #
    # hand_rot, hand_trans, hand_qpos = get_hand_params()
    # vis_img = tora_hand_visualizer.vis_grasp_pose(img, obj_pose, hand_rot, hand_trans, hand_qpos, cam_k)

    print("End")

