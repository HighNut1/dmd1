from os.path import join

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        alpha: "learning rate" of the centers. For each datapoint centers are updated by center + alpha * (datapoint - center)
    """

    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.01):
        super(CenterLoss, self).__init__()
        self.alpha = alpha
        
        # --- 修改点 1：适应端到端二值化 ---
        # 移除 L2 归一化，使用 {-1, 1} 的随机二值分布来初始化中心点
        initial_centers = torch.sign(torch.randn(num_classes, feat_dim))
        # 确保没有 0 出现 (防范极小概率的 randn 生成 0)
        initial_centers[initial_centers == 0] = 1.0 
        
        self.register_buffer(
            "centers",
            initial_centers,
            persistent=True,
        )
        # ---------------------------------
        
        self.counter = 0
        self.nupdate = 0

    def forward(self, x: torch.Tensor, labels: torch.LongTensor):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # Copy the centers into temporary matrix
        with torch.no_grad():
            batch_centers = torch.index_select(self.centers, 0, labels)
        
        # Get difference of x from batch centers
        diff = x - batch_centers
        
        # Update current centers
        # 中心点会在迭代中慢慢平滑，逐渐成为类内二值向量的均值（带有统计意义的浮点中心）
        with torch.no_grad():
            self.centers.index_add_(0, labels, diff, alpha=self.alpha)

        # --- 修改点 2：量纲缩放 ---
        # 因为 x 现在的绝对值都是 1，未经过 L2 归一化，导致 diff**2 随维度 D 线性放大。
        # 我们在这里除以特征维度 (x.shape[1])，将 Loss 缩放回原本代码 L2 归一化时的量级，
        # 从而完美复用 deep_print_loss.py 中预设的权重 W_CENTER_LOSS = 0.125
        # ---------------------------------
        return torch.sum(diff**2) / x.shape[1]