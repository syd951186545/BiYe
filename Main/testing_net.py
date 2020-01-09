import argparse
import pickle
import random
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

from BiYeSheJi.Module.environment import Environment
from BiYeSheJi.Script.buildNetworkGraph import get_relation_num, get_graph
from BiYeSheJi.Module.state_encoder import StateEncoder
from BiYeSheJi.Script.get_embedding import get_node_emb_matrix, get_relation_emb_dic
from BiYeSheJi.Configuration import config

# Hyper-parameters
seed = 1
render = False
Graph = get_graph()
num_episodes = 400000
env = Environment(Graph)
num_state = env.state_dim
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

    return matrix[torch.argmax(matrix, dim=0), torch.LongTensor(range(matrix.shape[1]))].view(1, matrix.shape[1])


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
        # action_prob = action_probilities.max()

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

    def analysis_state(self, state_tup_batch):
        """
        输入的是state_tuple 例如(2,3,4,6)表示的是当前状态，且当前状态是由2-3-4-转移至6节点
        修改：输入为 state_tuple 列表，输出为
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


if __name__ == '__main__':
    net = PolicyNetNet()
    data = net.analysis_state((3, 2, 4667, 502))
    net(data)
