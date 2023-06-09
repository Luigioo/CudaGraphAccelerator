import build.Debug.algo as algo

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Example usage
# G = nx.karate_club_graph()
G = nx.gnp_random_graph(50, 0.05)
G_dict = {node: list(G.neighbors(node)) for node in G.nodes()}
inipos = nx.random_layout(G)
newpos = algo.fruchterman_reingold_layout(G_dict, 50, 0.0, 1.0, 0.95, 0)
nx.draw(G, pos=newpos)
plt.show()

# print(algo.foo(1, 2))

# Visualize the result
# nx.draw(G, pos=pos)
# plt.show()