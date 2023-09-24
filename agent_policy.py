# Standard Library
from typing import Tuple

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party Library
import config

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
device = config.select_device


class AgentPolicy(nn.Module):
    def __init__(self, T, e, r, W, m) -> None:
        super().__init__()
        self.T = nn.Parameter(
            torch.tensor(T).float().to(device), requires_grad=True
        )
        self.e = nn.Parameter(
            torch.tensor(e).float().to(device), requires_grad=True
        )

        # 1 * Nの行列であることを想定する
        self.r = nn.Parameter(
            torch.tensor(r).float().view(-1, 1).to(device), requires_grad=True
        )

        self.W = nn.Parameter(
            torch.tensor(W).float().view(-1, 1).to(device), requires_grad=True
        )
        # self.m = nn.Parameter(
        #     torch.tensor(m).float().view(-1, 1).to(device), requires_grad=True
        # )

    def forward(
        self, attributes, edges, N
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edges = (edges > 0).float().to(device)

        tmp_tensor = self.W * torch.matmul(edges, attributes)
        # # 各列の最小値 (dim=0 は列方向)
        # min_values = torch.min(tmp_tensor, dim=0).values
        # # 各列の最大値 (dim=0 は列方向)
        # max_values = torch.max(tmp_tensor, dim=0).values

        # Min-Max スケーリング
        # tmp_tensor = (tmp_tensor - min_values) / (
        #     (max_values - min_values) + 1e-4
        # )

        # Computing feat
        feat = self.r * attributes + tmp_tensor * (1 - self.r)
        feat_prob = torch.tanh(feat)
        # # 各列の最小値 (dim=0 は列方向)
        # min_values = torch.min(feat, dim=0).values
        # # 各列の最大値 (dim=0 は列方向)
        # max_values = torch.max(feat, dim=0).values
        # Min-Max スケーリング
        # feat = (feat - min_values) / ((max_values - min_values) + 1e-4)

        # Feature normalization
        # norm = (
        #     feat.norm(dim=1)[:, None] + 1e-4
        # )  # add a small value to prevent division by zero
        # feat = torch.div(feat, norm)

        # Compute similarity
        x = torch.mm(feat, feat.t())
        # print(feat)
        x = torch.sigmoid(x.div(self.T).exp().mul(self.e))
        # print("prob", x)

        return x, feat, feat_prob

        # エッジの存在関数
        # x = torch.sigmoid(x)
        # x = torch.tanh(x)
        # x = torch.relu(x)
        # print("prob", x)
        # return x, feat

    def forward_neg(self, edges, feat):
        feat = feat.to(device)
        # 特徴量の正規化
        # norm = feat.norm(dim=1)[:, None] + 1e-8  # 零除算を防ぐための小さな値を加える
        # feat.div_(norm)
        # 類似度を計算
        x = torch.mm(feat, feat.t())  # 既にfeatはdeviceに存在
        x.neg_().add_(1.0)  # Negate and add in place

        # In-place演算を使用
        # rand = torch.rand_like(edges, device=device)  # 直接deviceで乱数生成
        # edges = (edges > 0).float().to(device)
        # edges.add_(rand.mul_(0.1))

        # x.mul_(edges)
        # # x = torch.clamp(x, min=0.0, max=1.0)  # Clip in place
        # x.clamp_(min=0.0)
        x.div_(self.T).exp_().mul_(self.e)  # exp and mul in place
        # x.div_(self.T).exp_()  # exp and mul in place
        x = torch.tanh(x)
        return x

    def predict(
        self, attributes, edges, N
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(attributes, edges, N)
