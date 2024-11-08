import os.path

import pyrealsense2 as rs
import numpy as np
import cv2
import shutil
import time
import trimesh


class RealsenseSaver(object):
    def __init__(self, s_root, s_ply=False):
        self.s_root = s_root
        self.s_ply = s_ply
        self.sub_dirs = ["colors", "depths"]
        self.check_s_root(s_root)

        self.cam_k = None

    def check_s_root(self, s_root):
        if os.path.exists(s_root):
            shutil.rmtree(s_root)
        os.mkdir(s_root)
        for sub_dir in self.sub_dirs:
            os.mkdir(os.path.join(s_root, sub_dir))

        if self.s_ply:
            os.mkdir(os.path.join(s_root, "plys"))

    def save_data(self, name, color, depth):
        s_color_path = os.path.join(self.s_root, "colors", name + "_color.jpg")
        s_depth_path = os.path.join(self.s_root, "depths", name + "_depth.npy")

        cv2.imwrite(s_color_path, color)
        np.save(s_depth_path, depth)

        if self.s_ply and self.cam_k is not None:
            pts, colors = self.image_and_depth_to_point_cloud(color, depth,  fx=self.cam_k[0, 0], fy=self.cam_k[1, 1],
                                                              cx=self.cam_k[0, 2], cy=self.cam_k[1, 2], max_depth=2.0)
            s_ply_path = os.path.join(self.s_root, "plys", name + "_pc.ply")
            self.save_2_ply(s_ply_path, pts[:, 0], pts[:, 1], pts[:, 2], colors.tolist())

    def save_2_ply(self, file_path, x, y, z, color=None):
        points = []
        if color == None:
            color = [[255, 255, 255]] * len(x)
        for X, Y, Z, C in zip(x, y, z, color):
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, C[2], C[1], C[0]))

        # for X, Y, Z, C in zip(x, y, z, color):
        #     points.append("%f %f %f %d %d %d 0\n" % (Z, X, Y, C[0], C[1], C[2]))

        file = open(file_path, "w")
        file.write('''ply
              format ascii 1.0
              element vertex %d
              property float x
              property float y
              property float z
              property uchar red
              property uchar green
              property uchar blue
              property uchar alpha
              end_header
              %s
              ''' % (len(points), "".join(points)))
        file.close()

    def image_and_depth_to_point_cloud(self, image, depth, fx, fy, cx, cy, max_depth=5.0):
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

    # def image_and_depth_to_point_cloud_with_intrinsics(self, image, depth, intrinsics_matrix):
    #     rows, cols = depth.shape
    #     u, v = np.meshgrid(range(cols), range(rows))
    #     uv_homogeneous = np.stack([u.flatten(), v.flatten(), np.ones_like(u.flatten())], axis=0)
    #     depth_flattened = depth.flatten()
    #     # 将深度为 0 或小于等于某个极小值的点标记为无效
    #     invalid_mask = (depth_flattened <= 0) | (depth_flattened < np.finfo(np.float32).eps)
    #     depth_valid = np.where(~invalid_mask, depth_flattened, 0)
    #     uv_homogeneous_valid = uv_homogeneous[:, ~invalid_mask]
    #     points_homogeneous = np.linalg.inv(intrinsics_matrix) @ uv_homogeneous_valid * depth_valid
    #     points = points_homogeneous[:3, :].T
    #     colors = image.reshape(-1, 3)[~invalid_mask]
    #     return points, colors

    def save_cam_k(self, cam_k):
        self.cam_k = cam_k
        s_cam_k_path = os.path.join(self.s_root, "cam_k.txt")
        np.savetxt(s_cam_k_path, cam_k)


def read_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # enable laser emitter or not
    depth_sensor = device.query_sensors()[0]
    emitter = depth_sensor.get_option(rs.option.emitter_enabled)
    print("emitter = ", emitter)

    set_emitter = 1
    depth_sensor.set_option(rs.option.emitter_enabled, set_emitter)
    emitter1 = depth_sensor.get_option(rs.option.emitter_enabled)
    print("new emitter = ", emitter1)

    device_product_line = str(device.get_info(rs.camera_info.product_line))
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    align_to = rs.stream.color
    align = rs.align(align_to)

    # get image instrinsices
    # profile_depth =profile.get_stream(rs.stream.depth)
    profile_color = profile.get_stream(rs.stream.color)
    intr = profile_color.as_video_stream_profile().get_intrinsics()
    cam_K_color = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])

    # 对于foundation-pose,需要得到第一帧的mesh模型和mask
    # mesh = trimesh.load(mesh_path)
    # mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)
    # cam_K = np.array([[615.37701416, 0., 313.68743896],
    #                   [0., 615.37701416, 259.01800537],
    #                   [0., 0., 1.]])

    count = 0
    # s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/dexgrasp_show_realsense_20241009"
    s_root = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/dexgrasp_show_realsense_202410091111"
    realsense_saver = RealsenseSaver(s_root, s_ply=1)
    realsense_saver.save_cam_k(cam_K_color)

    try:
        while True:
            t1 = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            count += 1
            if not aligned_depth_frame or not color_frame or count < 20:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data()) / 1e3
            color_image = np.asanyarray(color_frame.get_data())
            depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)

            H, W = color_image.shape[:2]
            color = cv2.resize(color_image, (W, H), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth_image_scaled, (W, H), interpolation=cv2.INTER_NEAREST)
            depth[(depth < 0.1) | (depth >= np.inf)] = 0

            realsense_saver.save_data(str(count), color[:, :, ::-1], depth)

            t2 = time.time() - t1
            print(count, int(1/t2), "fps")

            # cv2.imshow("depth", depth_image_scaled)
            # cv2.imshow("depth", depth)
            # cv2.waitKey(1)

            # 第一帧进行初始化
            # if i == 0:
            #     if len(mask.shape) == 3:
            #         for c in range(3):
            #             if mask[..., c].sum() > 0:
            #                 mask = mask[..., c]
            #                 break
            #     mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
            #     pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            # else:
            #     pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)

            # 可视化检测结果
            # center_pose = pose @ np.linalg.inv(to_origin)
            # vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
            # vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0,
            #                     is_input_rgb=True)
            # cv2.imshow('1', vis[..., ::-1])
            # cv2.waitKey(1)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # if count == 30:
            #     break
    finally:
        pipeline.stop()
        # cv2.destroyWindow()


def load_scale_mesh():
    # 加载从blender导出的mesh文件 并且进行scale
    mesh_file = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample/untitled.obj"
    mesh = trimesh.load(mesh_file, force='mesh')  # 从blender导出的模型需要设置force='mesh', 要不然没有vertices和vertex_normals这些属性
    mesh.apply_scale(0.001)  # mesh模型的单位是毫米，需要将其设置为米的单位 深度图输入的单位也是米
    mesh.export("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample/untitled2.obj")


if __name__ == "__main__":
    print("STart")
    read_realsense()
    # load_scale_mesh()
    print("end")