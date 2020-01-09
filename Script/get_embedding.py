import json
import torch
from BiYeSheJi.Script.buildNetworkGraph import get_relation_num
from BiYeSheJi.Configuration import config


def get_node_emb_matrix():
    relation_num = get_relation_num()
    with open(config.node_encode_filename, "r") as f:
        head = f.readline().split(" ")
        node_emb = torch.zeros(int(head[0]) + relation_num, int(head[1]))
        for line in f.readlines():
            line = line.split(" ")
            line_id = int(line[0])
            line_emb = torch.FloatTensor([float(line[x]) for x in range(1, len(line))])
            node_emb[line_id] = line_emb
    return node_emb


def get_relation_emb_dic():
    with open(config.edge_encode_filename, "r") as f:
        relation_emb_dic = json.load(f)
    return relation_emb_dic
