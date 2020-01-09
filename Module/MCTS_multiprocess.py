import json
import sys
import math
import random
import numpy as np
from BiYeSheJi.Script.buildNetworkGraph import get_graph

GRAPH = get_graph()
AVAILABLE_CHOICES = [1, -1, 2, -2]
AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
MAX_ROUND_NUMBER = 10


def select_best_child(node, is_exploration):  # 若子节点都扩展完了，求UCB值最大的子节点
    best_score = -sys.maxsize
    best_sub_node = None
    for sub_node in node.get_children():
        if is_exploration:
            # C = 1 / math.sqrt(2.0)  # C越大越偏向于广度搜索，越小越偏向于深度搜索
            C = 10
        else:
            C = 0.0
        # UCT:
        # left = sub_node.get_quality_value() / sub_node.get_visit_times()
        # right = math.log(node.get_visit_times()) / sub_node.get_visit_times()
        # score = left + C * math.sqrt(right)

        # UCT--
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        right = math.sqrt(node.get_visit_times()) / sub_node.get_visit_times()
        score = left + C * right

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


def select_node_to_simulate(node):  # 选择子节点的策略
    """
    选择扩展节点的策略，如果当前节点的有子节点未被扩展过，则选择一个扩展。
    若全部扩展过，就选择最优节点（PUCT策略选择）
    :param node:
    :return:
    """
    while not node.get_state().is_terminal():
        if node.is_all_expand():
            node = select_best_child(node, True)
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


def MCTS_main(node):  # 蒙特卡洛树搜索总函数
    computation_budget = 1000  # 模拟的最大次数
    for i in range(computation_budget):
        expend_node = select_node_to_simulate(node)
        reward = simulate(expend_node)
        # lock.acquire()
        backup(expend_node, reward)
        # lock.release()
    best_next_node = select_best_child(node, True)
    return best_next_node


def MCTS(start_node, target_node, query, file_i):
    state_0 = State()
    state_0.set_current_node(start_node)
    state_0.set_current_neighbor(GRAPH[start_node])
    state_0.set_cumulative_choices([query, start_node])
    state_0.target_node = target_node

    root = Node()
    root.set_state(state_0)
    print(root)
    PATH_LIST = []
    # node_t 是经历一次蒙特卡洛树搜索后根据UCT选择的最好下一节点
    for e in range(1000):
        path = [root.state.state_tup, ]
        node_t = root
        reward = 0
        while node_t.state.current_node != target_node:
            node_t = MCTS_main(node_t)
            if node_t is not None:
                reward = node_t.state.compute_reward()
                path.append(node_t.state.current_node)
                path.append(node_t.state.state_tup)
            else:
                break
        path.append(-1)
        path.append(path[-2])
        path.append(reward)
        PATH_LIST.append(path)

    with open(config.pathlist_file_dir + str(file_i) + ".json", "w") as pf:
        json.dump(PATH_LIST, pf)


if __name__ == '__main__':
    random.seed(1)
    from BiYeSheJi.Configuration import config
    from BiYeSheJi.Script.buildNetworkGraph import get_entity_dic, get_relation_dic
    from multiprocessing import Pool

    entity_dic = get_entity_dic()
    relation_dic = get_relation_dic()

    # train AthletePlaysForTeam data
    train_date = []
    with open(config.dataSet + "/train.txt", "r") as df:
        for i, line in enumerate(df.readlines()):
            line = line.replace("\n", "")
            line = line.split("\t")
            start_node = entity_dic[line[0]]
            target_node = entity_dic[line[2]]
            query = relation_dic[line[1]]
            if start_node not in GRAPH.nodes or target_node not in GRAPH.nodes:
                i -= 1
                continue
            train_date.append([start_node, target_node, query])

    with Pool(4) as p:
        for j, data in enumerate(train_date):
            p.apply_async(MCTS, args=(data[0], data[1], data[2], j,))
        p.close()
        p.join()