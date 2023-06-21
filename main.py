import build.Debug.algo as algo

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

def group_elements(arr):
    result = {}
    for i in range(0, len(arr), 2):
        # Get the two elements
        element1 = arr[i]
        element2 = arr[i+1] if i+1 < len(arr) else None

        result[i//2] = [element1, element2]

    return result

# Example usage
# G = nx.karate_club_graph()
numNodes = 500

G = nx.gnp_random_graph(numNodes, 0.05)

G_dict = {node: list(G.neighbors(node)) for node in G.nodes()}

# convert the 2d array G.edges, which has a stucture:
# [[v1, v2],[v1,v3],[v4,v5],...[v45, v48]]
# to: [v1, v2, v1, v3...v45, v48]
edge_edge_array = np.array(list(G.edges)).flatten()

start_time_cuda = time.time()
result = algo.fr_cuda(edge_edge_array, numNodes)
end_time_cuda = time.time()

cuda_time = end_time_cuda - start_time_cuda

cuda_pos = group_elements(result)


start_time_fr = time.time()
normal_pos = algo.fr(G_dict, 50, 0.0, 1.0, 0.95, 42)
end_time_fr = time.time()

fr_time = end_time_fr - start_time_fr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.set_title('Graph 1 (fr_cuda)\nTime: {:.4f} seconds'.format(cuda_time))
nx.draw(G, pos=cuda_pos, ax=ax1)

ax2.set_title('Graph 2 (fr)\nTime: {:.4f} seconds'.format(fr_time))
nx.draw(G, pos=normal_pos, ax=ax2)

plt.tight_layout()
plt.show()