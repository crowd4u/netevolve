# Standard Library
import gc
import os
from typing import List

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# First Party Library
import config
from agent import Agent
from agent_policy import AgentPolicy
from env import Env
from init_real_data import init_real_data

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
print(torch.__config__.parallel_info())
episodes = 32
story_count = 32
generate_count = 5
device = config.select_device

nodes: List[Agent] = []
LEARNED_TIME = 4
GENERATE_TIME = 5
TOTAL_TIME = 10

lr = 0.1
p_gamma = 0.8
attrs = []


def execute_data() -> None:
    np_alpha = []
    np_beta = []
    np_gamma = []
    np_delta = []

    with open("model.param.data.fast", "r") as f:
        lines = f.readlines()
        for index, line in enumerate(
            tqdm(lines, desc="load data", postfix="range", ncols=80)
        ):
            datus = line[:-1].split(",")
            np_alpha.append(np.float32(datus[0]))
            np_beta.append(np.float32(datus[1]))
            np_gamma.append(np.float32(datus[2]))
            np_delta.append(np.float32(datus[3]))

    # Define parameters of policy function
    T = np.array(
        [1.0 for i in range(len(np_alpha))],
        dtype=np.float32,
    )
    e = np.array(
        [1.0 for i in range(len(np_beta))],
        dtype=np.float32,
    )
    r = np.array(
        [1.0 for i in range(len(np_alpha))],
        dtype=np.float32,
    )
    w = np.array(
        [1e-0 for i in range(len(np_alpha))],
        dtype=np.float32,
    )
    m = np.array(
        [1e-2 for i in range(len(np_alpha))],
        dtype=np.float32,
    )
    # r = np.array(
    #     [1.0 for i in range(len(np_alpha))],
    #     dtype=np.float32,
    # )
    # w = np.array(
    #     [1e-2 for i in range(len(np_alpha))],
    #     dtype=np.float32,
    # )
    # m = np.array(
    #     [1e-8 for i in range(len(np_alpha))],
    #     dtype=np.float32,
    # )

    # Define parameters of reward Function
    alpha = torch.from_numpy(
        np.array(
            np_alpha,
            dtype=np.float32,
        ),
    ).to(device)

    beta = torch.from_numpy(
        np.array(
            np_beta,
            dtype=np.float32,
        ),
    ).to(device)

    gamma = torch.from_numpy(
        np.array(
            np_gamma,
            dtype=np.float32,
        ),
    ).to(device)

    delta = torch.from_numpy(
        np.array(
            np_delta,
            dtype=np.float32,
        )
    ).to(device)

    agent_policy = AgentPolicy(r=r, W=w, T=T, e=e, m=m)
    agent_optimizer = optim.Adadelta(agent_policy.parameters(), lr=lr)

    N = len(np_alpha)
    del np_alpha, np_beta, np_gamma, np_delta

    """_summary_
    setup data
    """
    load_data = init_real_data()

    field = Env(
        edges=load_data.adj[LEARNED_TIME].clone(),
        feature=load_data.feature[LEARNED_TIME].clone(),
        temper=T,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    memory = []
    for episode in tqdm(
        range(episodes), desc="episode", postfix="range", ncols=100
    ):
        if episode == 0:
            field.reset(
                load_data.adj[LEARNED_TIME].clone(),
                load_data.feature[LEARNED_TIME].clone(),
            )

        total_reward = 0
        for i in tqdm(
            range(story_count), desc="story", postfix="range", ncols=100
        ):
            memory = []
            reward = 0
            neighbor_state, feat = field.state()

            action_probs, predict_feat, _ = agent_policy.predict(
                edges=neighbor_state, attributes=feat, N=N
            )

            # field.update_attributes(predict_feat.detach())
            # reward = field.step(action_probs.detach().clone())
            reward = field.future_step(
                action_probs.detach().clone(), predict_feat.detach()
            )

            total_reward += reward

            memory.append((reward, action_probs))

        if not memory:
            continue
        G, loss = 0, 0
        for reward, prob in reversed(memory):
            G = reward + p_gamma * G
            loss += -torch.sum(torch.log(prob) * G)
        agent_optimizer.zero_grad()

        loss.backward()
        del loss

        agent_optimizer.step()

    gc.collect()

    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))

    for count in range(10):
        field.reset(
            load_data.adj[LEARNED_TIME].clone(),
            load_data.feature[LEARNED_TIME].clone(),
        )

        for t in range(TOTAL_TIME - GENERATE_TIME):
            gc.collect()
            neighbor_state, feat = field.state()

            action_probs, predict_feat, attr_probs = agent_policy.predict(
                edges=neighbor_state, attributes=feat, N=N
            )
            del neighbor_state, feat

            # field.update_attributes(predict_feat)
            # reward = field.step(action_probs)
            # reward = field.future_step(action_probs, predict_feat)
            reward = field.future_step(action_probs, predict_feat)

            target_prob = torch.ravel(predict_feat).to("cpu")
            del attr_probs
            gc.collect()
            detach_attr = (
                torch.ravel(load_data.feature[GENERATE_TIME + t])
                .detach()
                .to("cpu")
            )
            detach_attr[detach_attr > 0] = 1.0
            pos_attr = detach_attr.numpy()
            attr_numpy = np.concatenate([pos_attr], 0)
            target_prob = target_prob.to("cpu").detach().numpy()

            attr_predict_probs = np.concatenate([target_prob], 0)
            try:
                # NLLを計算
                criterion = nn.CrossEntropyLoss()
                error_attr = criterion(
                    torch.from_numpy(attr_predict_probs),
                    torch.from_numpy(attr_numpy),
                )
                auc_actv = roc_auc_score(attr_numpy, attr_predict_probs)
            except ValueError as ve:
                print(ve)
                pass
            finally:
                print("attr auc, t={}:".format(t), auc_actv)
                print("attr nll, t={}:".format(t), error_attr.item())
                attr_calc_log[count][t] = auc_actv
                attr_calc_nll_log[count][t] = error_attr.item()
            del (
                target_prob,
                pos_attr,
                attr_numpy,
                attr_predict_probs,
                auc_actv,
            )
            gc.collect()

            target_prob = torch.ravel(action_probs).to("cpu")
            del action_probs
            gc.collect()
            detach_edge = (
                torch.ravel(load_data.adj[GENERATE_TIME + t])
                .detach()
                .to("cpu")
            )
            pos_edge = detach_edge.numpy()
            edge_numpy = np.concatenate([pos_edge], 0)
            target_prob = target_prob.to("cpu").detach().numpy()

            edge_predict_probs = np.concatenate([target_prob], 0)

            try:
                # NLLを計算
                criterion = nn.CrossEntropyLoss()
                error_edge = criterion(
                    torch.from_numpy(edge_predict_probs),
                    torch.from_numpy(edge_numpy),
                )
                auc_actv = roc_auc_score(edge_numpy, edge_predict_probs)
            except ValueError as ve:
                print(ve)
                pass
            finally:
                print("-------")
                print("edge auc, t={}:".format(t), auc_actv)
                print("edge nll, t={}:".format(t), error_edge.item())
                print("-------")

                calc_log[count][t] = auc_actv
                calc_nll_log[count][t] = error_edge.item()
        print("---")

    np.save("proposed_edge_auc", calc_log)
    np.save("proposed_edge_nll", calc_nll_log)
    np.save("proposed_attr_auc", attr_calc_log)
    np.save("proposed_attr_nll", attr_calc_nll_log)


if __name__ == "__main__":
    execute_data()
