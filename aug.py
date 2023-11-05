import numpy as np
import torch
import torch.nn as nn
import tsaug
# np.random.randint(10,size=(10,3))
def DataTransform(sample, config):
    if "comb" in config.data_aug_mode:
        aug2 = overturn(sample, config.overturn_max_segments, config.overturn_ratio, config.overturn_use_partial,
                        config.overturn_use_noise, config.overturn_sigma)
        aug3, arange_mask = combination_permutation(sample, config.comb_replace_num, config.comb_min_ts,
                                       config.comb_max_ts, config.comb_use_rs, config.comb_use_per,
                                       config.comb_use_noise, config.comb_sigma)
        ## weak aug
        # aug1 = scaling(sample, sigma=config.jitter_scale_ratio)
        aug1 = shift(sample, config.shift_max_crop_len, config.shift_use_scaling, config.shift_sigma)
        # aug2 = scaling(sample, sigma=config.jitter_scale_ratio)
        return aug1, aug2, (aug3,arange_mask)
    else:
        # ## weak aug
        aug1 = overturn(sample, config.overturn_max_segments, config.overturn_ratio, config.overturn_use_partial,
                        config.overturn_use_noise, config.overturn_sigma)
        aug2 = shift(sample, config.shift_max_crop_len, config.shift_use_scaling, config.shift_sigma)

        # aug1 = scaling(sample, sigma=config.jitter_scale_ratio)
        # ## strong aug
        # aug2 = jitter(permutation(sample, max_segments=config.max_seg),config.jitter_ratio)
        # aug2= jitter(sample, config.jitter_ratio)
        # aug1 = jitter(sample, config.jitter_ratio)
        # aug1 = time_drift(sample)
        # aug2 = time_wrap(sample)
        return aug1, aug2


def time_wrap(data):
    data = data.detach().cpu().numpy()
    data = torch.FloatTensor(data)
    data = data.permute(0, 2, 1)
    data = data.cpu().numpy()
    data_aug = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(data)
    data_aug = torch.FloatTensor(data_aug).permute(0, 2, 1)
    return data_aug


def time_drift(data):
    # da@t@a@
    # data = data.detach().cpu().numpy()

    data = torch.FloatTensor(data)
    data = data.permute(0, 2, 1)
    data = data.cpu().numpy()
    data_aug = tsaug.Drift(max_drift=0.7, n_drift_points=5).augment(data)
    data_aug = torch.FloatTensor(data_aug).permute(0, 2, 1)
    return data_aug


def time_quantize(data):
    # data = torch.FloatTensor(data)
    data = data.permute(0, 2, 1)
    data = data.cpu().numpy()
    data_aug = tsaug.Quantize(n_levels=20).augment(data)
    data_aug = torch.FloatTensor(data_aug).permute(0, 2, 1)
    return data_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])


            u  =  np.random.permutation(splits)

            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


## 随机挑选x端 每段长度为t，随机替换为别的样本，考虑重排序（段重排）或者添加噪声
def combination_permutation(x, replace_num: int = 3, min_time_step: int = 1, max_time_step: int = 4, use_random_seg: bool = True,
                            use_permutation: bool = True, use_noise: bool = False, sigma:float=0.8):
    orig_steps = np.arange(x.shape[2])

    seq_len = x.shape[2]
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().numpy()
    ret = np.zeros_like(x)
    arange_mask = np.zeros(shape=(x.shape[0],seq_len))
    if use_random_seg:
        num_segs = np.random.randint(2, replace_num+1, size=(x.shape[0]))
    else:
        num_segs = np.ones(shape=(x.shape[0]))*(replace_num+1)
    for i, pat in enumerate(x):
        num_seg_t = int(num_segs[i])
        idx_all = list(range(x.shape[0]))
        idx_all.remove(i)
        choice_sample = np.random.choice(idx_all, num_seg_t-1, replace=False)
        split_points = np.random.choice(x.shape[2]-min_time_step, num_seg_t-1, replace=False)
        split_points.sort()
        random_comb_idx = np.random.choice(num_seg_t, num_seg_t-1, replace=False)
        crop_seq_len = np.random.randint(min_time_step, max_time_step, size=(num_seg_t-1))
        zj = pat
        arang_t = torch.ones(size=(1,seq_len))
        for j, s_idx in enumerate(choice_sample):
            comb_idx = random_comb_idx[j]
            if comb_idx == 0 :
                begin = 0
                end = split_points[comb_idx]
            elif comb_idx == num_seg_t-1:
                begin = split_points[comb_idx-1]
                end = min(seq_len, begin+crop_seq_len[j])
            else:
                begin = split_points[comb_idx-1]
                end = begin+crop_seq_len[j]
            zj[:, begin:end] = x[s_idx, :, begin:end]
            arang_t[:,begin:end] = s_idx
        if use_permutation:
            split_points.sort()
            splits = np.split(orig_steps, split_points)
            z = np.random.permutation(splits)
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            zj = zj[:, warp]
        ret[i] = zj
        arange_mask[i] = arang_t
    if use_noise:
         ret += np.random.normal(loc=0., scale=sigma, size=x.shape)
    return ret, arange_mask


def overturn(x, max_segments:int = 5, overturn_ratio: float = 0.5,
             use_partial: bool = True, use_noise: bool = True, sigma: float = 0.8):
    orig_steps = np.arange(x.shape[2])
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    ret = np.zeros_like(x)
    # num_segs = np.random.randint(50, max_segments, size=(x.shape[0]))
    # num_segs = np.random.randint(5, max_segments, size=(x.shape[0]))
    num_segs = np.random.randint(2, max_segments, size=(x.shape[0]))
    # num_segs = np.random.randint(32, max_segments, size=(x.shape[0]))
    for i, pat in enumerate(x):
        num_segs_t = num_segs[i]
        overturn_idx = list(range(num_segs_t))
        if use_partial:
            num_overturn = int(num_segs[i]*overturn_ratio)
            overturn_idx = np.random.choice(num_segs_t, num_overturn, replace=False)
        split_points = np.random.choice(x.shape[2] - num_segs_t, num_segs[i]-1, replace=False)
        split_points.sort()
        splits = np.split(orig_steps, split_points)
        for j in range(num_segs_t):
            if j in overturn_idx:
                splits[j]=splits[j][::-1]
        warp = np.concatenate(splits).ravel()
        ret[i] = pat[:, warp]
    if use_noise:
        ret = jitter(ret, sigma)
    return ret


def shift(x, max_crop_len: int = 32, use_scaling: bool = False, sigma: float = 1.1):
    seq_len = x.shape[2]
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    ret = np.zeros_like(x)
    # crop_size = np.random.randint(120, max_crop_len, size=(x.shape[0]))
    # crop_size = np.random.randint(4, max_crop_len, size=(x.shape[0]))
    # crop_size = np.random.randint(6, max_crop_len, size=(x.shape[0]))
    # crop_size = np.random.randint(4, max_crop_len, size=(x.shape[0]))
    crop_size = np.random.randint(6, max_crop_len, size=(x.shape[0]))
    # crop_size = np.random.randint(320, max_crop_len, size=(x.shape[0]))
    for i, pat in enumerate(x):
        crop_size_t = crop_size[i]
        left = np.arange(crop_size_t)
        right = list(range(crop_size_t, seq_len))
        right.extend(left)
        ret[i] = pat[:, right]
    if use_scaling:
        ret = scaling(ret, sigma)
    return ret


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

def generate_binomial_mask_3d(B, T, D, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)


def generate_lognormal_mask_3d(B,T,D,mu:float=0,sigma=1):
    return torch.from_numpy(np.random.normal(0,1,size=(B,T,D))).to(torch.bool)


def masked(x, mask_mode: str = "dim", mask_type:str = "mask_last"):
    if "dim" in mask_mode:
        if mask_type == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask_type == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask_type == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask_type == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask_type == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
    elif "time" in mask_mode:
        if mask_type == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask_type == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask_type == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask_type == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask_type == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
    elif "3dmask" in mask_mode:
        mask = generate_binomial_mask_3d(x.size(0), x.size(1), x.size(2)).to(x.device)

    else:
        mask = generate_lognormal_mask_3d(x.size(0), x.size(1), x.size(2)).to(x.device)
    return mask


if __name__ == "__main__":
    x = torch.randn(size=(2000, 9, 128))
    # combination_permutation(x)
    # overturn(x)
    # permutation(x)
    masked(x)

