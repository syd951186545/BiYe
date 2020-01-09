import json
from BiYeSheJi.Configuration import config
import networkx as nx

# 实体及关系编号
with open(config.dataSet + "/vocab/relation_vocab.json") as relationFile:
    relationEmb = json.load(relationFile, encoding="utf-8")
with open(config.dataSet + "/vocab/entity_vocab.json") as entityFile:
    entityEmb = json.load(entityFile, encoding="utf-8")


def get_relation_num():
    return len(relationEmb)


def get_entity_num():
    return len(entityEmb)


def get_entity_dic():
    return entityEmb


def get_relation_dic():
    return relationEmb


# 编号反向对应实体及关系
#
#
# .......................

# 构建知识图谱网络（有向图）
def get_graph():
    graph = nx.DiGraph()
    with open(config.dataSet + "/graph.txt") as graphFile:
        for line in graphFile:
            line = line.replace("\n", "")
            line = line.split("\t")
            if config.query != line[1]:
                graph.add_edge(entityEmb[line[0]], entityEmb[line[2]], relation=relationEmb[line[1]])

    return graph
