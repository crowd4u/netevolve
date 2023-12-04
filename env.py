# Standard Library
import math
from typing import Any

# Third Party Library
import networkx as nx
import numpy as np
import scipy
import torch

# First Party Library
import config

device = config.select_device


class Env:
    def __init__(self, edges, feature, temper, alpha, beta, gamma) -> None:
        self.edges = edges
        self.feature = feature.to(device)
        self.temper = temper
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # 特徴量の正規化
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature = self.feature.div_(norm)
        self.feature_t = self.feature.t()

        return

    def reset(self, edges, attributes) -> None:
        self.edges = edges
        self.feature = attributes
        # 特徴量の正規化
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature = self.feature.div(norm)
        self.feature_t = self.feature.t()

    def future_step(self, edges, attributes):
        # 特徴量の正規化
        norm = attributes.norm(dim=1)[:, None] + 1e-8
        attributes = attributes.div(norm)
        reward = (
            torch.sum(
                torch.softmax(torch.abs(attributes - self.feature), dim=1),
                dim=1,
            )
            * self.gamma
        )
        self.feature = attributes
        # 特徴量の正規化
        # norm = self.feature.norm(dim=1)[:, None] + 1e-8
        # self.feature = self.feature.div(norm)
        self.feature_t = self.feature.t()
        next_mat = edges.bernoulli()
        dot_product = torch.mm(self.feature, self.feature_t)
        reward = reward.add(next_mat.mul(dot_product).mul(self.alpha))
        # reward = next_mat.mul(dot_product).mul(self.alpha)
        costs = next_mat.mul(self.beta)
        reward = reward.sub(costs)
        # reward = reward.sum()
        self.edges = next_mat

        return reward.sum()

    def step(self, actions) -> Any:
        next_mat = actions.bernoulli()
        dot_product = torch.mm(self.feature, self.feature_t)
        reward = next_mat.mul(dot_product).mul(self.alpha)
        costs = next_mat.mul(self.beta)
        reward = reward.sub(costs)
        # reward = reward.sum()
        self.edges = next_mat

        return reward.sum()

    def update_attributes(self, attributes) -> None:
        self.feature = attributes
        # 特徴量の正規化
        norm = self.feature.norm(dim=1)[:, None] + 1e-8
        self.feature = self.feature.div(norm)
        self.feature_t = self.feature.t()

    def state(self) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor_mat = torch.mul(self.edges, self.edges)
        return neighbor_mat, self.feature
