import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    # L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source_size = source.size(0)
    target_size = target.size(0)
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)

    XX = torch.mean(kernels[:source_size, :source_size])
    YY = torch.mean(kernels[source_size:, source_size:])
    XY = torch.mean(kernels[:source_size, source_size:source_size + target_size])
    YX = torch.mean(kernels[source_size:source_size + target_size, :source_size])

    loss = torch.mean(XX + YY - XY - YX)
    # loss = XX  + YY - XY - YX
    return loss

def CORAL(source, target, DEVICE):
    eps=1e-5
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(DEVICE) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1 + eps)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1 + eps)

    # frobenius norm
    loss = (cs - ct).pow(2).sum()
    loss = loss / (4 * d * d)

    return loss
import torch
import torch.nn.functional as F
import numpy as np

def convert_to_onehot(sca_label, class_num=14):
    """Convert 1D integer labels to a one-hot NumPy array of shape (N, class_num)."""
    sca_label = sca_label.astype(int)
    return np.eye(class_num)[sca_label]

def cal_weight(
    s_label_or_logits: torch.Tensor,
    t_label_or_logits: torch.Tensor,
    source_size: int,
    target_size: int,
    class_num: int = 14
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the LMMD re-weighting matrices.  Handles:
     - s_label_or_logits: either (Ns,) int labels or (Ns, C) logits/probs
     - t_label_or_logits: either (Nt,) int labels or (Nt, C) logits/probs
    """
    # --- 1) Turn source into (Ns, C) prob-matrix and 1-D preds ---
    if s_label_or_logits.dim() == 1:
        # already integer labels
        s_idx = s_label_or_logits.detach().cpu().numpy().astype(int)
        s_vec = convert_to_onehot(s_idx, class_num=class_num)
    else:
        # logits or probs: make probs & preds
        s_logits = s_label_or_logits.detach()
        s_probs = F.softmax(s_logits, dim=1).cpu().numpy()
        s_idx   = s_probs.argmax(axis=1)
        s_vec   = s_probs

    # --- 2) Same for target ---
    if t_label_or_logits.dim() == 1:
        t_idx = t_label_or_logits.detach().cpu().numpy().astype(int)
        t_vec = convert_to_onehot(t_idx, class_num=class_num)
    else:
        t_logits = t_label_or_logits.detach()
        # if single-logit binary, cat to two probs
        if t_logits.size(1) == 1:
            t_logits = torch.cat([1 - t_logits, t_logits], dim=1)
        t_probs = F.softmax(t_logits, dim=1).cpu().numpy()
        t_idx   = t_probs.argmax(axis=1)
        t_vec   = t_probs

    # --- 3) Normalize each column to sum=1 (avoid zeros) ---
    s_sum = s_vec.sum(axis=0, keepdims=True)
    s_sum[s_sum == 0] = 1
    s_vec /= s_sum

    t_sum = t_vec.sum(axis=0, keepdims=True)
    t_sum[t_sum == 0] = 1
    t_vec /= t_sum

    # --- 4) Find intersecting classes ---
    intersect = np.intersect1d(np.unique(s_idx), np.unique(t_idx)).astype(int)

    # --- 5) Build masks and apply ---
    mask_s = np.zeros((source_size, class_num), dtype=np.float32)
    mask_s[:, intersect] = 1
    s_vec *= mask_s

    mask_t = np.zeros((target_size, class_num), dtype=np.float32)
    mask_t[:, intersect] = 1
    t_vec *= mask_t

    # --- 6) Compute pairwise weights ---
    if intersect.size > 0:
        w_ss = (s_vec @ s_vec.T) / intersect.size
        w_tt = (t_vec @ t_vec.T) / intersect.size
        w_st = (s_vec @ t_vec.T) / intersect.size
    else:
        w_ss = np.zeros((1,), dtype=np.float32)
        w_tt = np.zeros((1,), dtype=np.float32)
        w_st = np.zeros((1,), dtype=np.float32)

    # --- 7) Convert back to CUDA tensors ---
    return (
        torch.from_numpy(w_ss).cuda(),
        torch.from_numpy(w_tt).cuda(),
        torch.from_numpy(w_st).cuda()
    )

def lmmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    s_label_or_logits: torch.Tensor,
    t_label_or_logits: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None
) -> torch.Tensor:
    """
    LMMD loss between source and target embeddings.
    Args:
      source, target        : (Ns,D), (Nt,D)
      s_label_or_logits     : (Ns,) ints or (Ns,C) logits/probs
      t_label_or_logits     : (Nt,) ints or (Nt,C) logits/probs
    """
    Ns, Nt = source.size(0), target.size(0)

    # Compute re-weight matrices
    w_ss, w_tt, w_st = cal_weight(
        s_label_or_logits, t_label_or_logits,
        source_size=Ns, target_size=Nt,
        class_num=(t_label_or_logits.size(1)
                   if t_label_or_logits.dim()>1 else int(s_label_or_logits.max())+1)
    )

    # Build kernel matrix (you need your own gaussian_kernel)
    kernels = gaussian_kernel(
        source, target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma
    ).cuda()

    SS = kernels[:Ns, :Ns]
    TT = kernels[Ns:, Ns:]
    ST = kernels[:Ns, Ns:Ns+Nt]

    loss = torch.sum(w_ss * SS) + torch.sum(w_tt * TT) - 2 * torch.sum(w_st * ST)
    return loss


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res