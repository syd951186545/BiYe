import sys
import math
import random
import torch

import numpy as np
from torch import nn

from BiYeSheJi.Script.buildNetworkGraph import get_graph
from BiYeSheJi.Script.get_embedding import get_node_emb_matrix, get_relation_emb_dic
from BiYeSheJi.Configuration import config

GRAPH = get_graph()
AVAILABLE_CHOICES = [1, -1, 2, -2]
AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
MAX_ROUND_NUMBER = 10
node_emb_matrix = get_node_emb_matrix()
edge_emb_dic = get_relation_emb_dic()
Graph = get_graph()


def _Max(matrix):
    """
    取矩阵每一列最大值得到一行向量
    :param matrix:
    :return:
    """

    return matrix[torch.argmax(matrix, dim=0), torch.LongTensor(range(matrix.shape[1]))].view(1, matrix.shape[1])


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.GRUCell = nn.GRUCell(3 * 128, 404)
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

        return torch.max(Qsa_value)


def best_child(node, is_exploration):  # 若子节点都扩展完了，求UCB值最大的子节点
    best_score = -sys.maxsize
    best_sub_node = None
    for sub_node in node.get_children():
        if is_exploration:
            C = 1 / math.sqrt(2.0)  # C越大越偏向于广度搜索，越小越偏向于深度搜索
        else:
            C = 0.0
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        right = math.log(node.get_visit_times()) / sub_node.get_visit_times()
        score = left + C * math.sqrt(right)
        if score > best_score:
            best_score = score
            best_sub_node = sub_node
        if score == best_score:
            best_sub_node = random.choice([best_sub_node, sub_node])
    return best_sub_node


def expand(node):  # 得到未扩展的子节点
    tried_sub_node_id = [sub_node.get_state().current_node for sub_node in node.get_children()]
    new_state = node.get_state().get_next_state_with_random_choice_without_all_expended(tried_sub_node_id, node.parent)
    # while new_state in tried_sub_node_states:  # 可能造成无限循环
    #     new_state = node.get_state().get_next_state_with_random_choice_without_all_expended()
    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    return sub_node


def tree_policy(node):  # 选择子节点的策略
    """
    选择扩展节点的策略，如果当前节点的有子节点未被扩展过，则选择一个扩展。
    若全部扩展过，就选择最优节点扩展（PUCT策略选择）
    :param node:
    :return:
    """
    while not node.get_state().is_terminal():
        if node.is_all_expand():
            sub_node = best_child(node, True)
        else:
            sub_node = expand(node)
        return sub_node
    return node


# 模拟该节点，获得最终可能回报
def default_policy(node):
    current_state = node.get_state()
    simulation = 0
    while not current_state.is_terminal():
        current_state = current_state.get_next_state_with_random_choice(current_state.node_parent)
        simulation += 1
        if simulation > 20:
            break
    final_state_reward = current_state.compute_reward()
    return final_state_reward


def backup(node, reward):
    while node is not None:
        node.quality_value_add_n(reward)
        node.visit_times_add_one()
        node = node.parent


class Node(object):
    def __init__(self):
        self.parent = None
        self.children = []
        self.visit_times = 0
        self.quality_value = 0
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def set_parent(self, parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def set_children(self, children):
        self.children = children

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        # 算模拟的总平均值作为节点的q
        # self.quality_value = self.quality_value + (n - self.quality_value) / (self.visit_times + 1)
        self.quality_value += n

    def is_all_expand(self):
        if self.parent:
            if len(self.children) == len(GRAPH[self.state.current_node]):
                return True
            else:
                return False
        else:
            if len(self.children) == len(GRAPH[self.state.current_node]) + 1:
                return True
            else:
                return False

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def __repr__(self):
        if self.visit_times != 0:
            return "Node:{} avgQ:{} N:{}".format(self.state.current_node, self.quality_value,
                                                 self.visit_times)
        else:
            return "Node:{} Q:{} N:{}".format(self.state.current_node, self.quality_value, self.visit_times)


class State(object):  # 某游戏的状态，例如模拟一个数相加等于1的游戏
    def __init__(self):
        self.current_node = None  # 当前状态，网络节点编号,-1表示终止节点，无实意
        self.candidate_actions = []  # 当前状态，网络节点邻接节点（包含对应边）
        self.current_round_index = 0  # 第几轮网络节点选择
        self.target_node = None
        self.node_parent = None

        self.state_tup = []  # 选择过程记录,选择网络节点路径

    def is_terminal(self):  # 判断游戏是否结束
        if self.current_round_index == (MAX_ROUND_NUMBER - 1) or self.current_node == -1:
            return True
        else:
            return False

    def compute_reward(self, stopcode=0):
        # 模拟终止局面时的得分，当选择终止节点时父节点是目标节点则应该给+1reward
        # 划掉？？？但排除其直接从“初始节点-(训练关系)-目标节点”，等同于训练数据？？？
        # if self.current_node == -1:
        #     return 1 if self.node_parent == self.target_node else Qvalue_net(
        #         Qvalue_net.analysis_state(self.state_tup[:-1])).detach()
        # else:
        #     return Qvalue_net(Qvalue_net.analysis_state(self.state_tup)).detach()
        if self.current_node == -1:
            return 1 if self.node_parent == self.target_node else 0
        else:
            return 1 if self.current_node == self.target_node else 0

    def set_current_node(self, value):
        self.current_node = value

    def set_node_parent(self, value):
        self.node_parent = value

    def set_candidate_action(self, value):
        self.candidate_actions = value

    def set_current_round_index(self, round):
        self.current_round_index = round

    def set_state_tup(self, choices):
        self.state_tup = choices

    def get_next_state_with_random_choice(self, parent):  # 得到下个状态
        actions = [keys for keys in self.candidate_actions.keys()]
        actions.append(-1)
        if parent in actions:
            actions.remove(parent)
        random_choice = random.choice(actions)
        if random_choice == -1:
            next_state = State()
            next_state.set_current_node(-1)
            next_state.set_current_round_index(self.current_round_index + 1)
            next_state.set_state_tup(self.state_tup + [random_choice])
            next_state.set_node_parent(self.current_node)
            next_state.target_node = self.target_node
            return next_state
        next_state = State()
        next_state.set_current_node(random_choice)
        next_state.set_current_round_index(self.current_round_index + 1)
        next_state.set_candidate_action(GRAPH[random_choice])
        next_state.set_state_tup(self.state_tup + [random_choice])
        next_state.set_node_parent(self.current_node)
        next_state.target_node = self.target_node
        return next_state

    def get_next_state_with_random_choice_without_all_expended(self, tried_child, parent):
        actions = [keys for keys in self.candidate_actions.keys()]
        actions.append(-1)
        actions = list(set(actions).difference(set(tried_child)))
        if parent in actions:
            actions.remove(parent)

        random_choice = random.choice(actions)
        if random_choice == -1:
            next_state = State()
            next_state.set_current_node(-1)
            next_state.set_current_round_index(self.current_round_index + 1)
            next_state.set_state_tup(self.state_tup + [random_choice])
            next_state.set_node_parent(self.current_node)
            next_state.target_node = self.target_node
            return next_state
        next_state = State()
        next_state.set_current_node(random_choice)
        next_state.set_current_round_index(self.current_round_index + 1)
        next_state.set_candidate_action(GRAPH[random_choice])
        next_state.set_state_tup(self.state_tup + [random_choice])
        next_state.set_node_parent(self.current_node)
        next_state.target_node = self.target_node
        return next_state


def MCTS_main(node):  # 蒙特卡洛树搜索总函数
    computation_budget = 100  # 模拟的最大次数
    for i in range(computation_budget):
        expend_node = tree_policy(node)
        reward = default_policy(expend_node)
        backup(expend_node, reward)
    best_next_node = best_child(node, True)
    return best_next_node


if __name__ == '__main__':
    torch.manual_seed(1)
    start_node = 2
    target_node = 3  # 训练数据
    target_query = 4
    Qvalue_net = QNet()

    state_0 = State()
    state_0.set_current_node(start_node)
    state_0.set_candidate_action(GRAPH[start_node])
    state_0.set_state_tup([target_query, start_node])
    state_0.target_node = target_node

    root = Node()
    root.set_state(state_0)
    print(root)

    # node_t 是经历一次蒙特卡洛树搜索后根据UCT选择的最好下一节点
    for e in range(100):
        path = [root.state.state_tup, ]
        node_t = MCTS_main(root)

        while node_t is not None:
            if node_t.state.current_node == -1:
                print(node_t.state.compute_reward())
                path.append(-1)
                path.append("ST")
            else:
                print(node_t.state.compute_reward())
                path.append(node_t.state.current_node)
                path.append(node_t.state.state_tup)
            node_t = MCTS_main(node_t)
        print(path)
