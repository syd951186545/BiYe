import math
import os
import random
import sys
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

root_dir = os.path.abspath('.')
sys.path.append(root_dir)
from Script.buildNetworkGraph import get_relation_num, get_graph
from Script.get_embedding import get_node_emb_matrix, get_relation_emb_dic
from Configuration import config

GRAPH = get_graph()
MAX_ROUND_NUMBER = 10


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
        self.quality_value += n

    def is_all_expand(self):
        # 包含了终止动作 -1
        if self.parent is not None:
            if len(self.children) == len(self.state.candidate_actions) - 1:
                return True
            else:
                return False
        else:
            if len(self.children) == len(self.state.candidate_actions):
                return True
            else:
                return False

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def __repr__(self):
        return "Node:{},Q/N:{}/{}".format(self.state.current_node, self.quality_value, self.visit_times)


class State(object):  # 某游戏的状态，例如模拟一个数相加等于1的游戏
    def __init__(self):
        self.current_node = None  # 当前状态，网络节点编号,-1表示终止节点，无实意
        self.candidate_actions = []  # 当前状态，网络节点邻接节点（包含对应边）
        self.current_round_index = 0  # 第几轮网络节点选择
        self.target_node = None
        self.node_parent = None

        self.state_tup = []  # 选择过程记录,选择网络节点路径

    def is_terminal(self):  # 判断游戏是否结束
        if self.current_round_index == (MAX_ROUND_NUMBER - 1) or self.current_node == self.target_node \
                or list(self.candidate_actions.keys()) == [self.node_parent]:
            return True
        else:
            return False

    def compute_reward(self):
        # 模拟终止局面时的得分，当选择终止节点时是目标节点则应该给+1 reward
        return 1 if self.current_node == self.target_node else 0

    def set_current_node(self, value):
        self.current_node = value

    def set_node_parent(self, value):
        self.node_parent = value

    def set_current_neighbor(self, value):
        self.candidate_actions = value

    def set_current_round_index(self, round):
        self.current_round_index = round

    def set_cumulative_choices(self, choices):
        self.state_tup = choices

    def get_next_state_with_random_choice(self, parent):  # 得到下个状态
        actions = [keys for keys in self.candidate_actions.keys()]
        if parent in actions:
            actions.remove(parent)
        if not actions:
            return self
        else:
            random_choice = random.choice(actions)

            next_state = State()
            next_state.set_current_node(random_choice)
            next_state.set_current_round_index(self.current_round_index + 1)
            next_state.set_current_neighbor(GRAPH[random_choice])
            next_state.set_cumulative_choices(self.state_tup + [random_choice])
            next_state.set_node_parent(self.current_node)
            next_state.target_node = self.target_node
        return next_state

    def get_next_state_with_random_choice_without_all_expended(self, tried_child, parent):
        actions = [keys for keys in self.candidate_actions.keys()]
        actions = list(set(actions).difference(set(tried_child)))
        if parent in actions:
            actions.remove(parent)

        random_choice = random.choice(actions)
        next_state = State()
        next_state.set_current_node(random_choice)
        next_state.set_current_round_index(self.current_round_index + 1)
        next_state.set_current_neighbor(GRAPH[random_choice])
        next_state.set_cumulative_choices(self.state_tup + [random_choice])
        next_state.set_node_parent(self.current_node)
        next_state.target_node = self.target_node
        return next_state


def PUCT(node, Policy_net, is_exploration):  # 若子节点都扩展完了，求UCB值最大的子节点
    best_score = -sys.maxsize
    best_sub_node = None
    state = Policy_net.analysis_state(node.get_state().state_tup)
    action_candidate, Qsa_values = Policy_net(state)

    for sub_node in node.get_children():
        if is_exploration:
            # C = 1 / math.sqrt(2.0)  # C越大越偏向于广度搜索，越小越偏向于深度搜索
            C = 10
        else:
            C = 0.0

        # PUCT:
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        right = math.sqrt(node.get_visit_times()) / sub_node.get_visit_times()

        Qsa = Qsa_values[0][action_candidate.index(sub_node.state.current_node)]
        score = left + C * Qsa * right

        # UCT:
        # left = sub_node.get_quality_value() / sub_node.get_visit_times()
        # right = math.log(node.get_visit_times()) / sub_node.get_visit_times()
        # score = left + C * math.sqrt(right)

        # UCT--
        # left = sub_node.get_quality_value() / sub_node.get_visit_times()
        # right = math.sqrt(node.get_visit_times()) / sub_node.get_visit_times()
        # score = left + C * right

        if score > best_score:
            best_score = score
            best_sub_node = sub_node
        if score == best_score:
            best_sub_node = random.choice([best_sub_node, sub_node])
    return best_sub_node


def expand_child(node):  # 得到未扩展的子节点
    tried_sub_node_id = [sub_node.get_state().current_node for sub_node in node.get_children()]
    new_state = node.get_state().get_next_state_with_random_choice_without_all_expended(tried_sub_node_id,
                                                                                        node.state.node_parent)
    # while new_state in tried_sub_node_states:  # 可能造成无限循环
    #     new_state = node.get_state().get_next_state_with_random_choice_without_all_expended()
    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    return sub_node


def select_node_to_simulate(node, Policy_net):  # 选择子节点的策略
    """
    选择扩展节点的策略，如果当前节点的有子节点未被扩展过，则选择一个扩展。
    若全部扩展过，就选择最优节点（PUCT策略选择）
    :param Policy_net:
    :param node:
    :return:
    """
    while not node.get_state().is_terminal():
        if node.is_all_expand():
            node = PUCT(node, Policy_net, True)
        else:
            sub_node = expand_child(node)
            return sub_node
    return node


def simulate(node):
    """
    模拟该节点，获得最终可能回报
    终止条件： 1.到达目标节点 2.仿真10步长（状态自身计数） 3.没有可选动作
    仿真策略：随机选择下一动作（节点），但排除父节点
    仿真回报：到达目标节点+1， 否则为0
    :param node:
    :return:
    """
    current_state = node.get_state()

    while not current_state.is_terminal():
        current_state = current_state.get_next_state_with_random_choice(current_state.node_parent)

    final_state_reward = current_state.compute_reward()
    return final_state_reward


def backup(node, reward):
    while node is not None:
        node.visit_times_add_one()
        node.quality_value_add_n(reward)
        node = node.parent


def MCTS_main(node):  # 蒙特卡洛树搜索总函数
    computation_budget = 1000  # 模拟的最大次数
    for i in range(computation_budget):
        expend_node = select_node_to_simulate(node)
        reward = simulate(expend_node)
        backup(expend_node, reward)
    best_next_node = PUCT(node, True)
    return best_next_node


def Policy_MCTS(node, Policy_net):
    computation_budget = 100  # 模拟的最大次数
    for i in range(computation_budget):
        expend_node = select_node_to_simulate(node, Policy_net)
        reward = simulate(expend_node)
        backup(expend_node, reward)
    best_next_node = PUCT(node, Policy_net, True)
    return best_next_node


# Hyper-parameters
seed = 1
render = False
Graph = get_graph()
num_episodes = 40000

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
        self.FA.weight.data.normal_(0, 0.1)

        self.FS = nn.Linear(404, 128)
        self.FS.weight.data.normal_(0, 0.1)

        self.FP1 = nn.Linear(128 + 128, 1)
        self.FP2 = nn.Linear(128 + 128, 1)
        self.FP1.weight.data.normal_(0, 0.1)
        self.FP2.weight.data.normal_(0, 0.1)

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
            hnts_matrix = F.relu(hnts_matrix)
            hAt_ = _Max(hnts_matrix)
            hat_ = hnts_matrix[action_id - node_index1 - 1].view(1, -1)
            qt = self.GRUCell(torch.cat((hAt_, hat_, dataX[node_index2]), 1), qt)
        hSt = self.FS(qt)
        hSt = F.relu(hSt)
        # q1 = hA0*q0; q2 = hA1*q1; hS2 = f(q2) ,所以需要再求hA2
        neighbor_matrix = torch.cat(tuple([dataX[j] for j in range(self.node_index[-1] + 1, len(dataX))]))
        hnts_matrix = self.FA(neighbor_matrix)
        hnts_matrix = F.relu(hnts_matrix)

        hAt = _Max(hnts_matrix)

        #
        u0 = self.FP1(torch.cat((hSt, hAt), 1))
        u0 = F.relu(u0)

        hSt_ks = hSt.expand(hnts_matrix.shape)
        # uks = hSt.mm(hnts_matrix.t())
        uks = self.FP2(torch.cat((hSt_ks, hnts_matrix), 1))
        uks = F.relu(uks)

        # return Q
        # Qsa_value = self.sigmoid(torch.cat((u0, uks), 1))
        Qsa_value = torch.cat((u0, uks.t()), 1)

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

    def PUCT(self, node):

        # MCTS->action->next_node

        return MCTS_main(node, self.act_net)

    def store_transition(self, path):
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
                losses = torch.tensor(0).float()
                for i, j in enumerate(index):
                    # state = self.target_net.analysis_state(mem.state)
                    mem = self.memory[j]
                    reward = torch.tensor(mem.reward).float()
                    action = mem.action
                    if action == -1:
                        target_v = reward
                    else:
                        with torch.no_grad():
                            # Doule DQN
                            next_state = self.act_net.analysis_state(mem.next_state)
                            action_candidate, Qsa_values = self.act_net(next_state)
                            action2 = action_candidate[Qsa_values.max(1)[1]]

                            next_state = self.target_net.analysis_state(mem.next_state)
                            action_candidate2, Qsa_values2 = self.target_net(next_state)
                            Qsa2 = Qsa_values2[0][action_candidate2.index(action2)]
                            target_v = reward + self.gamma * Qsa2

                            # Nature DQN
                            # target_v = reward + self.gamma * self.target_net(next_state)[1].max()
                    state = self.act_net.analysis_state(mem.state)
                    action_candidate, Qsa_values = self.act_net(state)
                    Qsa = Qsa_values[0][action_candidate.index(action)]

                    loss = self.loss_func(target_v, Qsa)
                    losses += loss
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', losses / self.batch_size, self.update_count)
                self.update_count += 1
                if self.update_count % 500 == 0:
                    self.target_net.load_state_dict(self.act_net.state_dict())
                    torch.save(self.act_net.state_dict(), config.act_net_model_dir + str(self.update_count) + ".model")


from BiYeSheJi.Module.environment import Environment

env = Environment(Graph)
env.init_TreeList(config.dataSet + "/train.txt")


def main():
    agentP = DQN()
    find_target = 0
    for i_ep in range(num_episodes):
        root = env.reset()
        if render: env.render()
        # node_t 是经历一次蒙特卡洛树搜索后根据UCT选择的最好下一节点
        path = [root.state.state_tup, ]
        node_t = root
        reward = 0
        while node_t.state.current_node != root.state.target_node:
            node_t = Policy_MCTS(node_t, agentP.act_net)
            if node_t is not None:
                reward = node_t.state.compute_reward()
                path.append(node_t.state.current_node)
                path.append(node_t.state.state_tup)
            else:
                break
        env.update(root)  # 更新root的q和 v的值
        path.append(-1)
        path.append(path[-2])
        path.append(reward)
        agentP.store_transition(path)

        find_target += reward
        agentP.writer.add_scalar('find_target/step', find_target, global_step=i_ep)
        agentP.update()


if __name__ == '__main__':
    main()
