import json
import random
import torch

from BiYeSheJi.Script.buildNetworkGraph import get_graph, get_relation_num
from BiYeSheJi.Module.state_encoder import StateEncoder
from BiYeSheJi.Configuration import config


def set_seed(seed=None):
    random.seed(seed)
    return


NODE_DIM = config.node_encode_dim
EDGE_DIM = get_relation_num()
INPUT_SIZE, HIDDEN_SIZE = NODE_DIM + EDGE_DIM, EDGE_DIM


class Environment:
    def __init__(self, Graph):
        self.reward_range = (0, 1)
        torch.manual_seed(1)

        # 环境获取的状态的维度128+404（邻接点和边被压缩成128表示，404维度是历史信息维度）
        self.state_dim = NODE_DIM + EDGE_DIM

        # 读取self.Graph = nx.read_gpickle("./data/DBLP_labeled.Graph")
        # 或者直接获取返回值buildNetworkGraph.get_graph()
        self.Graph = Graph
        self.Graph = get_graph()

        # 初始化状态编码器
        self.StateEncoder = StateEncoder()

        # 初始化一些环境属性，空值
        self.path = []
        self.current_node = None
        self.target_node = None
        self.query = None

        self.qt = None
        self.st = None
        self.done = False

        self.TreeDic = {}

    def reset(self, start_node_id, target_node_id, query_id):

        # q0 = GRU(query,[0.,0.,ns]) ，dim = 404
        # qt = GRU(qt-1,[HAt,Hat,nt])
        # HAt表示邻接节点和边总的编码 dim=128
        # st = qt∪HAt
        q0 = self.StateEncoder.get_q0(start_node_id, query_id)
        HAt = self.StateEncoder.get_Neighbor_encode(self.Graph[start_node_id])
        s0 = torch.cat((q0, HAt), 1)

        self.current_node = start_node_id
        self.target_node = target_node_id
        self.path = [start_node_id, ]

        self.qt = q0
        self.st = s0
        self.done = False

    def step(self, action_node_id, use_MCTS=config.use_MCTS):
        info = self.path
        if use_MCTS:

        else:
            if -1 == action_node_id:
                reward = 1 if self.target_node == self.current_node else 0
                self.done = True
            else:
                qt_ = self.qt
                HAt, Hat = self.StateEncoder.get_Neighbor_encode(self.Graph[self.current_node], action_node=action_node_id)
                nt = self.StateEncoder.get_node_emb(action_node_id)
                input_ = torch.cat((HAt, Hat, nt), 1)
                self.qt = self.StateEncoder.GRUCell(input_, qt_).detach()
                self.st = torch.cat((self.qt, HAt), 1)

                self.current_node = action_node_id
                self.path.append(action_node_id)
                reward = 0
                self.done = False

        # return next_state, reward, self.done, info
        return self.st, reward, self.done, info

    def render(self):
        print(self.path)

    def BaseReward(self, action):
        return
