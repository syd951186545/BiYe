import os

root_dir = os.path.dirname(os.path.abspath('.'))

#  dataSet 使用的数据集在项目中的位置 root_dir为项目当前绝对地址
dataSet = root_dir + "/DataSets/AthletePlaysForTeam"

# 查询关系语句(在制作GRAPH-net时剔除)
query = "concept:athleteplaysforteam"

#  MCTS 是否采用蒙特卡洛树搜索
use_MCTS = True

# graph-net中的编码方式,one-hot,node2vec,
node_encode = "node2vec"
node_encode_dim = 128
edge_encode = "one-hot"
# embedding 存放目录
node_encode_filename = root_dir + "/Embeddings/01/node2vec.emb"
edge_encode_filename = root_dir + "/Embeddings/01/edge2emb.json"
# embedding模型存放目录
node_model_filename = root_dir + "/Embeddings/01/node2vec.model"
edge_model_filename = root_dir + "/Embeddings/01/edge2vec.model"

# 训练用的path list 存放路径
pathlist_file_dir = root_dir + "/TrainData/AthletePlaysForTeam1000/"

# 模型的可视化Summary 存放路径
Summary_dir = root_dir+"/Model/summary/"
# 模型参数 存放路径
act_net_model_dir = root_dir+"/Model/actnet_model/"
