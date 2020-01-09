import random

from BiYeSheJi.Module.environment import Environment
from BiYeSheJi.Script.buildNetworkGraph import get_graph
import torch
import json
random.seed(1)
start_node_id, target_node_id, query_id = 2, 3, 4
env = Environment(get_graph())
env.reset(start_node_id, target_node_id, query_id)
print(env.st)
for i in range(5):
    action_node_id = random.choice(list(env.Graph[env.current_node].keys()))
    env.step(action_node_id)
    print(env.current_node)
    print(env.path)
    print(env.st)
