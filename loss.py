import math
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import time
from evaluation import *
from torch.nn.functional import normalize



class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l


        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy

    def Structure_guided_Contrastive_Loss(self, h_i, h_j, S):
        S_1 = S.repeat(2, 2)
        all_one = torch.ones(h_i.size(0)*2, h_i.size(0)*2).to('cuda')
        S_2 = all_one - S_1
        N = 2 * h_i.size(0)
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim1 = torch.multiply(sim, S_2)
        sim_i_j = torch.diag(sim, h_i.size(0))
        sim_j_i = torch.diag(sim, -h_i.size(0))
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim1[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss



def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def mutual_information_Loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss


def cosine_similarity_loss(prototypes_view1, prototypes_view2):
    # 计算余弦相似度损失
    similarity_loss = 0.0
    for prototype_view1 in prototypes_view1:
        for prototype_view2 in prototypes_view2:
            similarity = cosine_similarity([prototype_view1], [prototype_view2])[0, 0]
            similarity_loss += similarity
    return similarity_loss

class Proto_Align_Loss(nn.Module):
    def __init__(self):
        super(Proto_Align_Loss, self).__init__()

    def forward(self, gt, P):
        mse = nn.MSELoss()
        Loss1 = mse(gt, P)

        return Loss1


class Instance_Align_Loss(nn.Module):
    def __init__(self):
        super(Instance_Align_Loss, self).__init__()

    def forward(self, gt, P):
        mse = nn.MSELoss()
        Loss2 = mse(gt, P)

        return Loss2



class DeepMVCLoss(nn.Module):
    def __init__(self, num_samples, num_clusters):
        super(DeepMVCLoss, self).__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()

        return mask

    def forward_prob(self, q_i, q_j):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = (p_i * torch.log(p_i)).sum()

        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = (p_j * torch.log(p_j)).sum()

        entropy = ne_i + ne_j

        return entropy

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):

        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.num_clusters
        q = torch.cat((q_i, q_j), dim=0)

        if normalized:
            sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
        else:
            sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)

        sim_i_j = torch.diag(sim, self.num_clusters)
        sim_j_i = torch.diag(sim, -self.num_clusters)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


def guassian_kernel_mmd(source, target, kernel_mul=2, kernel_num=4, fix_sigma=None):
    """Gram kernel matrix
    source: sample_size_1 * feature_size
    target: sample_size_2 * feature_size
    kernel_mul: bandwith of kernels
    kernel_num: number of kernels
    return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)
            [ K_ss K_st
              K_ts K_tt ]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def DDA(source, target, kernel_mul=2, kernel_num=4, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel_mmd(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, m:]
    YX = kernels[m:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt,Target<->Target

    loss = XX.sum() + XY.sum() + YX.sum() + YY.sum()
    return loss

# def compute_joint(view1, view2):
#     """Compute the joint probability matrix P"""
#
#     bn, k = view1.size()
#     assert (view2.size(0) == bn and view2.size(1) == k)
#
#     p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
#     p_i_j = p_i_j.sum(dim=0)
#     p_i_j = (p_i_j + p_i_j.t()) / 2.   # symmetrise
#     p_i_j = p_i_j / p_i_j.sum()        # normalise
#
#     return p_i_j


def MIA(view1, view2, lamb=10.0, EPS=sys.float_info.epsilon):
    """MMI loss"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb) * torch.log(p_j) \
                      - (lamb) * torch.log(p_i))

    loss = loss.sum()

    return loss
