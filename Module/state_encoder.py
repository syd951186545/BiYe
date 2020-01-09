from torch import nn
from BiYeSheJi.Script.buildNetworkGraph import get_relation_num
from BiYeSheJi.Script.get_embedding import get_node_emb_matrix, get_relation_emb_dic
from BiYeSheJi.Configuration import config
import torch

relation_dim = get_relation_num()
INPUT_SIZE_liner = relation_dim + 128
INPUT_SIZE_GRU = 3 * 128

OUT_SIZE_liner = 128
HIDDEN_SIZE_GRU = relation_dim


def _Max(matrix):
    """
    取矩阵每一列最大值得到一行向量
    :param matrix:
    :return:
    """

    return matrix[torch.argmax(matrix, dim=0), torch.LongTensor(range(matrix.shape[1]))].view(1, matrix.shape[1])


class StateEncoder:
    def __init__(self):
        self.FA = nn.Linear(INPUT_SIZE_liner, OUT_SIZE_liner)
        self.GRUCell = nn.GRUCell(INPUT_SIZE_GRU, HIDDEN_SIZE_GRU)
        self.maxPool = nn.MaxPool1d(1, 1)
        self.node_emb_matrix = get_node_emb_matrix()
        self.edge_emb_dic = get_relation_emb_dic()

    def get_node_emb(self, action_node_id):
        return torch.FloatTensor(self.node_emb_matrix[action_node_id]).reshape(1, config.node_encode_dim)

    def get_q0(self, start_node_id, query_id):
        # 输入3*128和上一次记忆输出404，输出404
        return self.GRUCell(
            torch.cat(
                (torch.zeros((1, config.node_encode_dim)), torch.zeros((1, config.node_encode_dim)),
                 torch.FloatTensor(self.node_emb_matrix[start_node_id]).reshape(1, config.node_encode_dim))
                , 1),
            torch.FloatTensor(self.edge_emb_dic[str(query_id)]).reshape(1, relation_dim)).detach()

    def encodeNE(self, last_state, Neighbor_relation_matrix, nt):
        """
        :param Neighbor_relation_matrix: 邻居节点与边拼接的表示矩阵
        :param nt: 当前节点表示
        :param last_state: Hs（t-1），上一状态
        :return:
        """
        HAt = self.FA(Neighbor_relation_matrix)
        HAt_max = self.maxPool(HAt)
        current_input = torch.cat((last_state, nt, HAt_max), 1)
        current_state = self.GRUCell(current_input)
        return current_state

    def get_Neighbor_encode(self, Neighbour_dic, action_node=None):
        """
        :param action_node: 如果有策略选择动作，则返回动作的编码
        :param Neighbour_dic: 当前节点的邻接节点及边
        例如：{36394: {'relation': 187}, 2116: {'relation': 163}
        :return:HAt（邻接节点和边的状态编码） and Hat（各个动作的编码）
        """
        neighbors = [key for key in Neighbour_dic]
        Input = torch.zeros((len(neighbors), config.node_encode_dim + relation_dim))
        for i in range(len(neighbors)):
            edge_emb = self.edge_emb_dic[str(Neighbour_dic[neighbors[i]]["relation"])]
            node_emb = self.node_emb_matrix[neighbors[i]]
        Input[i] = torch.cat((torch.FloatTensor(node_emb).reshape(1, config.node_encode_dim),
                              torch.FloatTensor(edge_emb).reshape(1, relation_dim)), 1)
        Output = self.FA(Input).detach()
        HAt = _Max(Output)
        if action_node:
            # 动作node的编码
            return HAt, Output[neighbors.index(action_node)].view(1, config.node_encode_dim)

        return HAt
