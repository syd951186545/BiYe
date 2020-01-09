import argparse
import json
import pickle
import random
from collections import namedtuple
from itertools import count

import os, time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

root_dir = os.path.abspath('.')
# 存入当前目录
sys.path.append(root_dir)

from Script.buildNetworkGraph import get_relation_num, get_graph
from Script.get_embedding import get_node_emb_matrix, get_relation_emb_dic
from Configuration import config

# Hyper-parameters
seed = 1
render = False
Graph = get_graph()
num_episodes = 400000

num_action = 128
torch.manual_seed(seed)
random.seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

relation_dim = get_relation_num()
INPUT_SIZE_liner = relation_dim + 128
INPUT_SIZE_GRU = 3 * 128

OUT_SIZE_liner = 128
HIDDEN_SIZE_GRU = relation_dim

node_emb_matrix = get_node_emb_matrix()
edge_emb_dic = get_relation_emb_dic()


def _Max(matrix):
    """
    取矩阵每一列最大值得到一行向量
    :param matrix:
    :return:
    """

    return matrix[torch.argmax(matrix, dim=0), torch.LongTensor(range(matrix.shape[1]))].view(1, -1)


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.GRUCell = nn.GRUCell(INPUT_SIZE_GRU, HIDDEN_SIZE_GRU)
        self.FA = nn.Linear(532, 128)
        self.FS = nn.Linear(404, 128)
        self.FP = nn.Linear(128 + 128, 1)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.node_index = []  # 每个节点在data中的位置
        self.action_index = []  # 上一次的动作在data中的位置
        self.action_candidate = [-1, ]  # 最后一个节点的候选动作节点

    def analysis_state(self, state_tup):
        """
        输入的是state_tuple 例如(2,3,4,6)表示的是当前状态，且当前状态是由2-3-4-转移至6节点
        把当前表示状态转换成embedding作为net的输入，需要分别处理并记住哪些embedding输入什么编码结构中
        :param state_tup:
        :return:
        """
        query = torch.FloatTensor(edge_emb_dic[str(state_tup[0])]).view(1, -1)
        dataX = [query, ]
        node_index = []  # 标注节点在data中的位置,以及节点的下一个动作的位置
        last_action_index = []
        action_candidate = [-1, ]
        for i in range(1, len(state_tup)):
            node_index.append(len(dataX))
            dataX.append(node_emb_matrix[state_tup[i]].view(1, -1))
            for node, edge in Graph[state_tup[i]].items():
                node_emb = node_emb_matrix[node]
                if i < len(state_tup) - 1:
                    if node == state_tup[i + 1]:
                        last_action_index.append(len(dataX))
                relation_emb = edge_emb_dic[str(edge["relation"])]
                c = torch.cat((node_emb.view(1, -1), torch.FloatTensor(relation_emb).view(1, -1)), 1)
                dataX.append(c)
                if i == len(state_tup) - 1:  # 获取最后一个节点的候选动作节点
                    action_candidate.append(node)

        self.node_index = node_index
        self.action_index = last_action_index
        self.action_candidate = action_candidate

        return dataX

    def forward(self, dataX):
        # q0 = GRU([0.,0.,ns],query) ，dim = 404
        qt = self.GRUCell(
            torch.cat(
                (torch.zeros((1, config.node_encode_dim)),
                 torch.zeros((1, config.node_encode_dim)), dataX[1]), 1
            )
            , dataX[0])

        for i in range(len(self.node_index) - 1):
            node_index1 = self.node_index[i]
            node_index2 = self.node_index[i + 1]
            action_id = self.action_index[i]
            neighbor_matrix = torch.cat(tuple([dataX[j] for j in range(node_index1 + 1, node_index2)]))
            hnts_matrix = self.FA(neighbor_matrix)
            hAt_ = _Max(hnts_matrix)
            hat_ = hnts_matrix[action_id - node_index1 - 1].view(1, -1)
            qt = self.GRUCell(torch.cat((hAt_, hat_, dataX[node_index2]), 1), qt)
        hSt = self.FS(qt)
        # q1 = hA0*q0; q2 = hA1*q1; hS2 = f(q2) ,所以需要再求hA2
        neighbor_matrix = torch.cat(tuple([dataX[j] for j in range(self.node_index[-1] + 1, len(dataX))]))
        hnts_matrix = self.FA(neighbor_matrix)
        hAt = _Max(hnts_matrix)

        #
        u0 = self.FP(torch.cat((hSt, hAt), 1))
        uks = hSt.mm(hnts_matrix.t())

        # return action and prob
        action_probilities = self.softmax(torch.cat((u0, uks), 1))
        # action_node = self.action_candidate[action_probilities.argmax()]
        # action_prob = torch.max(action_probilities)

        return self.action_candidate, action_probilities


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.GRUCell = nn.GRUCell(INPUT_SIZE_GRU, HIDDEN_SIZE_GRU)
        self.FA = nn.Linear(532, 128)
        self.FS = nn.Linear(404, 128)
        self.FP = nn.Linear(128 + 128, 1)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.node_index = []  # 每个节点在data中的位置
        self.action_index = []  # 上一次的动作在data中的位置
        self.action_candidate = [-1, ]  # 最后一个节点的候选动作节点

    def analysis_state(self, state_tup):
        """
        输入的是state_tuple 例如(2,3,4,6)表示的是当前状态，且当前状态是由2-3-4-转移至6节点
        把当前表示状态转换成embedding作为net的输入，需要分别处理并记住哪些embedding输入什么编码结构中
        :param state_tup:
        :return:
        """
        query = torch.FloatTensor(edge_emb_dic[str(state_tup[0])]).view(1, -1)
        dataX = [query, ]
        node_index = []  # 标注节点在data中的位置,以及节点的下一个动作的位置
        last_action_index = []
        action_candidate = [-1, ]
        for i in range(1, len(state_tup)):
            node_index.append(len(dataX))
            dataX.append(node_emb_matrix[state_tup[i]].view(1, -1))
            for node, edge in Graph[state_tup[i]].items():
                node_emb = node_emb_matrix[node]
                if i < len(state_tup) - 1:
                    if node == state_tup[i + 1]:
                        last_action_index.append(len(dataX))
                relation_emb = edge_emb_dic[str(edge["relation"])]
                c = torch.cat((node_emb.view(1, -1), torch.FloatTensor(relation_emb).view(1, -1)), 1)
                dataX.append(c)
                if i == len(state_tup) - 1:  # 获取最后一个节点的候选动作节点
                    action_candidate.append(node)

        self.node_index = node_index
        self.action_index = last_action_index
        self.action_candidate = action_candidate

        return dataX

    def forward(self, dataX):
        # q0 = GRU([0.,0.,ns],query) ，dim = 404
        qt = self.GRUCell(
            torch.cat(
                (torch.zeros((1, config.node_encode_dim)),
                 torch.zeros((1, config.node_encode_dim)), dataX[1]), 1
            )
            , dataX[0])

        for i in range(len(self.node_index) - 1):
            node_index1 = self.node_index[i]
            node_index2 = self.node_index[i + 1]
            action_id = self.action_index[i]
            neighbor_matrix = torch.cat(tuple([dataX[j] for j in range(node_index1 + 1, node_index2)]))
            hnts_matrix = self.FA(neighbor_matrix)
            hAt_ = _Max(hnts_matrix)
            hat_ = hnts_matrix[action_id - node_index1 - 1].view(1, -1)
            qt = self.GRUCell(torch.cat((hAt_, hat_, dataX[node_index2]), 1), qt)
        hSt = self.FS(qt)
        # q1 = hA0*q0; q2 = hA1*q1; hS2 = f(q2) ,所以需要再求hA2
        neighbor_matrix = torch.cat(tuple([dataX[j] for j in range(self.node_index[-1] + 1, len(dataX))]))
        hnts_matrix = self.FA(neighbor_matrix)
        hAt = _Max(hnts_matrix)

        #
        u0 = self.FP(torch.cat((hSt, hAt), 1))
        uks = hSt.mm(hnts_matrix.t())
        # return Q
        Qsa_value = self.sigmoid(torch.cat((u0, uks), 1))

        return self.action_candidate, Qsa_value


class DQN:
    capacity = 2000
    learning_rate = 1e-3
    memory_count = 0
    batch_size = 100
    gamma = 0.995
    update_count = 0

    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.act_net = QNet(), QNet()
        self.memory = [None] * self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter(config.Summary_dir)

    def select_action(self, state):

        # MCTS->action

        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.act_net(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= 0.9:  # epslion greedy
            action = np.random.choice(range(num_action), 1).item()
        return action

    def store_transition(self, path_list):
        for path in path_list:
            reward = path[-1]
            for i in range(len(path) // 2 - 1):
                state = path[2 * i]
                action = path[2 * i + 1]
                next_state = path[2 * (i + 1)]
                transition = Transition(state, action, reward, next_state)

                index = self.memory_count % self.capacity
                self.memory[index] = transition
                self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self):
        print(self.memory_count)
        if self.memory_count < self.capacity:
            print("wait enough train date")
            return
        else:
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size,
                                      drop_last=False):
                losses = torch.zeros((len(index), 1))
                for i, j in enumerate(index):
                    # state = self.target_net.analysis_state(mem.state)
                    mem = self.memory[j]
                    action = mem.action
                    reward = torch.tensor(mem.reward).float().cuda()
                    next_state = self.target_net.module.analysis_state(mem.next_state).cuda()

                    with torch.no_grad():
                        target_v = reward + self.gamma * self.target_net(next_state)[1].max()

                    state = self.act_net.module.analysis_state(mem.state).cuda()
                    action_candidate, Qsa_values = self.act_net(state)
                    Qsa = Qsa_values[0][action_candidate.index(action)]

                    loss = self.loss_func(target_v, Qsa)
                    losses[i] = loss
                losses = torch.mean(losses)
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', losses, self.update_count)
                self.update_count += 1
                if self.update_count % 100 == 0:
                    self.target_net.load_state_dict(self.act_net.state_dict())


def main():
    agent = DQN()
    for i_ep in range(num_episodes):
        state = env.reset()
        if render: env.render()
        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if render: env.render()
            transition = Transition(state, action, reward, next_state)
            agent.store_transition(transition)
            state = next_state
            if done or t >= 9999:
                agent.writer.add_scalar('live/finish_step', t + 1, global_step=i_ep)
                agent.update()
                if i_ep % 10 == 0:
                    print("episodes {}, step is {} ".format(i_ep, t))
                break


if __name__ == '__main__':
    i = 0
    agent = DQN()
    # while os.path.exists(config.pathlist_file_dir + str(i) + ".json"):
    #     with open(config.pathlist_file_dir + str(i) + ".json") as paths_file:
    #         path_list = json.load(paths_file)
    #         agent.store_transition(path_list)
    #         agent.update()
    #     i += 1
    agent.target_net.analysis_state([4, 2, 4667, 66544])
    for i in range(52):
        if os.path.exists(config.pathlist_file_dir + str(i) + ".json"):
            with open(config.pathlist_file_dir + str(i) + ".json") as paths_file:
                path_list = json.load(paths_file)
                agent.store_transition(path_list)
                agent.update()
    torch.save(agent.act_net.state_dict(), config.act_net_model_dir)
