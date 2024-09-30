import torch


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



if __name__ == "__main__":
    print("Start")
    # debug_uniform_sample()
    debug_manual_sedd()
    print("End")
