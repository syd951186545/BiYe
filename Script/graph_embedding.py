from BiYeSheJi.Script import buildNetworkGraph
from BiYeSheJi.Script.node2vec import Node2Vec
from BiYeSheJi.Configuration import config
import numpy as np
import json

entity_num = buildNetworkGraph.get_entity_num()
relation_num = buildNetworkGraph.get_relation_num()
entity_dic = buildNetworkGraph.get_entity_dic()
relation_dic = buildNetworkGraph.get_relation_dic()

GRAPH = buildNetworkGraph.get_graph()


def node_embedding():
    if "node2vec" == config.node_encode:
        node2vec = Node2Vec(GRAPH, dimensions=config.node_encode_dim, walk_length=20, num_walks=20, workers=4)
        model = node2vec.fit(window=4, min_count=1, batch_words=4, workers=4)
        # Look for most similar nodes
        model.wv.most_similar('2')  # Output node names are always strings
        # Save embeddings for later use
        model.wv.save_word2vec_format(config.node_encode_filename)
        # Save model for later use
        model.save(config.node_model_filename)


def edge_embedding():
    if "one-hot" == config.edge_encode:
        eye = np.eye(relation_num)
        with open(config.edge_encode_filename, "w") as ef:

            # ef.write(str(relation_num) + " " + str(relation_num) + "\n")
            i = 0
            relation_emb_dic = {}
            for keys in relation_dic:
                relation_emb_dic[relation_dic[keys]] = list(eye[i])
                # ef.write("{0} {1}\n".format(str(relation_dic[keys]), str(eye[i])))

            json.dump(relation_emb_dic, ef)


if __name__ == '__main__':
    edge_embedding()
