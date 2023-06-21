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

def calculate_edge_crossings(graph, pos):
    crossings = 0

    # Iterate over each pair of edges
    for u, v in graph.edges():
        for x, y in graph.edges():
            if u != x and u != y and v != x and v != y:
                # Check for edge crossings
                if do_edges_cross(pos[u], pos[v], pos[x], pos[y]):
                    crossings += 1

    return crossings

def do_edges_cross(p1, p2, p3, p4):
    # Check if two edges cross each other
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    return (
        min(x1, x2) < max(x3, x4) and
        min(y3, y4) < max(y1, y2) and
        min(x3, x4) < max(x1, x2) and
        min(y1, y2) < max(y3, y4) and
        ((x1 - x2) * (y3 - y1) - (y1 - y2) * (x3 - x1)) * ((x1 - x2) * (y4 - y1) - (y1 - y2) * (x4 - x1)) < 0 and
        ((x3 - x4) * (y1 - y3) - (y3 - y4) * (x1 - x3)) * ((x3 - x4) * (y2 - y3) - (y3 - y4) * (x2 - x3)) < 0
    )

# Example usage
# G = nx.karate_club_graph()
numNodes = 5000

G = nx.gnp_random_graph(numNodes, 0.05, 42)

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

# print("edge crossings: "+str(calculate_edge_crossings(G, cuda_pos)))

edge_edge_array = np.array(list(G.edges)).flatten()
start_time_fr = time.time()
normal_pos = algo.fr(edge_edge_array, numNodes)
end_time_fr = time.time()
fr_time = end_time_fr - start_time_fr
normal_pos = group_elements(normal_pos)

# print("edge crossings: "+str(calculate_edge_crossings(G, normal_pos)))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.set_title('Graph 1 (fr_cuda)\nTime: {:.4f} seconds'.format(cuda_time))
nx.draw(G, pos=cuda_pos, ax=ax1)

ax2.set_title('Graph 2 (fr)\nTime: {:.4f} seconds'.format(fr_time))
nx.draw(G, pos=normal_pos, ax=ax2)

plt.tight_layout()
plt.show()