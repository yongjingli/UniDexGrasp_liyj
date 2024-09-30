# 1. AttributeError: module 'distutils' has no attribute 'version'
# pip install setuptools==56.1.0

# 2. AttributeError: 'NoneType' object has no attribute 'seek'. You can only torch.load from a file that is seekable. Please pre-load the data into a buffer like io.BytesIO and try to load from it instead.
# 预训练权重的准备
# https://github.com/PKU-EPIC/UniDexGrasp/issues/6       几个模型的权重，其中ipdf的权重是有问题的，需要替换为下面的权重
# https://github.com/PKU-EPIC/UniDexGrasp/issues/12      ipdf的权重的
# 权重的目录结构
# ➜  runs git:(liyj_dev) ✗ tree -L 3
# ├── exp_cm
# │└── ckpt
# │└── model_0006.pt
# ├── exp_glow
# │└── ckpt
# │└── model_1750.pt
# ├── exp_ipdf
# │└── ckpt
# │└── model_0500.pt


# 3. 对数据进行可视化需要result.pt，需要在eval的时候进行保存
# running / dex_generation / tests / visualize_result.py, I need result.pt,
# I have solved it. When you run the "eval.py", you should add a line of code in the end such as "torch.save(result, result.pt)" .
# 在main的底部加入
#     pt_path = log_dir + '/result.pt'
#     torch.save(result, pt_path)


# 4. eval的时候数据太多加载时间过程
# 将datasets里object_dataset.py里的加载数据减少
# incase too many
# self.object_list = self.object_list[2:5]

# 出现out of mem的情况
# 将eval_config里的tta的batchsize减少
#    batch_size: 50
#     batch_size: 10

# 5. 采用visualize result进行可视化
# --exp_dir为result.pt放置的位置，默认是在'eval'下
# --num 代表显示的是第几个

# 可视化的效果不是很好
# https://zhuanlan.zhihu.com/p/650320613 是否需要参考别人的训练权重

# 可视化训练过程 tensorboard --logdir=./
# issue 有关训练的问题，一般都是训练时间特别长
# https://github.com/PKU-EPIC/UniDexGrasp/issues/16
# https://github.com/PKU-EPIC/UniDexGrasp/issues/10

# 训练ipdf部分
# python ./network/train.py --config-name ipdf_config --exp-dir ./ipdf_train

# 训练cmnet部分
# python ./network/train.py --config-name cm_net_config --exp-dir ./cm_net_train

# 训练glow部分
# python ./network/train.py --config-name glow_config  --exp-dir ./glow_train

# 联合训练
# python ./network/train.py --config-name glow_joint_config --exp-dir ./glow_train


# 推理的时候将训练时的随机变量去除
# 在object dataset 中每次随机取pose
# indices = np.random.permutation(len(pose_matrices))[:cfg['n_samples']]
# 每次预测的结果都不一样, 去除随机因素
# indices = np.arange(len(pose_matrices))[:cfg['n_samples']]

# 在pointnetpp_encoder 中的fps的第一个indx是随机得到的
# idx = fps(pos, batch, ratio=self.ratio, random_start=False)
# idx = fps(pos, batch, ratio=self.ratio)

# 将object dataset 中随机采样table点云注释，发现将这个注释了效果会好比较多
# table_pc_extra = torch.stack([distances * torch.cos(theta), distances * torch.sin(theta), torch.zeros_like(distances)], dim=1)
# table_pc = torch.cat([table_pc, table_pc_extra])

# 将ipdf_network中的均匀采样部分进行替换，读取固定不变的均匀采样,发现对结果的影响也挺大的
# np.save("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/sample_point.npy", sample_point.detach().cpu().numpy())
# sample_point = torch.from_numpy(
#     np.load("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/sample_point.npy")).to(
#     device)

# 进一步发现，在训练时的配置文件用的num_worker是为1，将其修改为16，时间大幅度下降, 对于cmnet需要修改为64才能将一个epoch压缩到30分钟左右，因为期间需要动态生成hand_mesh，很耗时

# 对于数据结构
# mesh底下的coacd
# CoACD CoACD的核心是其独特的碰撞感知算法
# coacd.urdf 包含物体的urdf文件
# decomposed.obj 为物体完整的obj文件
# coacd_convex_piece_0.obj coacd_convex_piece_1.obj 是上述obj文件中各个部分


# 进行联合训练时, 发现一训练就会闪退
# for epoch in range(start_epoch, cfg["total_epoch"]):
# glow_joint_config.yaml中设置epoch 为2000,但是加载的start_epoch比这个还大，所以直接就退出了
# epoch的保存序号跟配置文件里的这个有关，需要仔细检查设置
# freq:
#     step_epoch: 100
#     save: 10000  # per iter
#     plot: 100  # per iter
#     test: 50000 # per iter


# 加载联合训练的权重时会出现权重不匹配的情况，需要重新指向联合训练的权重
# 在进行infer_unidexgrasp.py中进行联合训练的权重加载
# pose_trainer = get_trainer(cfg, 'pose', logger)
# pose_trainer = get_trainer_joint(cfg, 'pose', logger)

# 简直是大坑，在trainer.py中进行联合训练的时候，加载完参数后，居然进行了一次初始化，导致原来加载好的参数都不能用了
# self.scheduler = get_scheduler(self.optimizer, cfg)
# self.apply(weights_init(cfg['weight_init']))

# 修改为不是联合训练才进行参数的初始化
# if 'joint_training' in cfg['model']:
#     if not cfg['model']['joint_training']:
#         self.apply(weights_init(cfg['weight_init']))
# else:
#     self.apply(weights_init(cfg['weight_init']))

# 联合训练的脚本
# python ./network/train.py --config-name glow_joint_config_train.yaml --exp-dir ./runs/exp_joint_train/glow_train_batch128
# 将glow的权重指向 ./runs/exp_joint_train/glow_train_batch128

# 修改cm_net_config.yaml的权重位置(联合训练的时候默认读取这个配置文件)
# exp_dir: ./runs/exp_joint_train/cm_net_train_batch128

# 修改 vim ipdf_config.yaml的权重位置(联合训练的时候默认读取这个配置文件)
# exp_dir: ./runs/exp_joint_train/ipdf_train_batch128

# 修改configs/eval_config.yaml 将batchsize降低进行测评
# +#    batch_size: 50
# +    batch_size: 10