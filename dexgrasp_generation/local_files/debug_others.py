import torch
import numpy as np
from utils_data import save_2_ply
import matplotlib.cm as cm


def debug_uniform_sample():
    import torch
    from torch.distributions import Normal

    # 创建标准正态分布
    standard_normal = Normal(0, 1)

    # 假设我们有 num_samples 和 embedded_context
    num_samples = 5
    embedded_context = torch.randn(num_samples, 10)  # 示例上下文，形状为 (num_samples, 10)

    # 采样并计算对数概率
    sample, log_prob = standard_normal.sample_and_log_prob(
        sample_shape=(num_samples,), context=embedded_context
    )

    print("Sampled values:", sample)
    print("Log probabilities:", log_prob)


def debug_manual_sedd():

    for i in range(2):
        rand_b = torch.randn(5)
        rand_c = torch.randn(5)

        torch.manual_seed(233)
        rand_a = torch.randn(5)
        print(rand_a)


def save_cmnet_npy_2_ply():
    pts_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/infer_result/sem_Bottle-908e85e13c6fbde0a1ca08763d503f0e_points.npy"
    pts = np.load(pts_path)[0]

    s_ply_path = pts_path.replace(".npy", ".ply")

    cmmap_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/infer_result/sem_Bottle-908e85e13c6fbde0a1ca08763d503f0e_cm.npy"
    cm_map = np.load(cmmap_path)[0]
    # cm_map[3000:] = 0.5
    cm_map = 1 - cm_map

    vmin = 0
    vmax = 1
    color_map = cm.get_cmap('rainbow')
    d = np.clip(cm_map, vmin, vmax)
    d = d / vmax
    colors = color_map(d)
    colors = (colors[:, :3] * 255).astype(np.uint8)

    save_2_ply(s_ply_path, pts[:, 0], pts[:, 1], pts[:, 2], color=colors.tolist())


def check_pc_tables():
    pc_table_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/generate_dataset/DFCData/meshes/baojie/tide/pcs_table.npy"
    pc_table = np.load(pc_table_path)
    pc_table = pc_table[0]
    s_ply_path = pc_table_path.replace(".npy", ".ply")
    save_2_ply(s_ply_path, pc_table[:, 0], pc_table[:, 1], pc_table[:, 2],)


if __name__ == "__main__":
    print("Start")
    # debug_uniform_sample()
    # debug_manual_sedd()
    save_cmnet_npy_2_ply()
    # check_pc_tables()
    print("End")
