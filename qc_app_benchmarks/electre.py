import numpy as np
from pyDecision.algorithm import electre_iii, electre_i_v
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

def rec_rank(matrix, G, index, level, prev):
    def dfs(node, visited):
        visited.add(node)
        for idx, val in enumerate(matrix[node]):
            if val == 'R' and idx not in visited:
                dfs(idx, visited)
        return visited
    visited = dfs(index, set()) 
    last = prev[:]
    stop_counter = 0
    new_prev = prev[:]
    flag = 1
    number_of_p = level
    while (flag):
        for node in visited:
            if matrix[node].count('P-') == number_of_p:
                G.add_node(node, subset = level)
                new_prev.append(node)
                for i in prev:
                    if matrix[node][i] == "P-":
                        G.add_edge(i, node)
                        if i in new_prev:
                            new_prev.remove(i)
        if prev == new_prev:
            if stop_counter == 0:
                level_to_remember = level
            stop_counter += 1
        else:
            stop_counter = 0
            level += 1
        if stop_counter == len(matrix) + 1:
            flag = 0
        number_of_p += 1
        prev = new_prev[:]
    return G, prev, level_to_remember

def create_ranking(matrix):
    G = nx.DiGraph()
    prev = []
    level = 0
    while level < len(matrix):
        best = []
        for j in range(len(matrix)):
            if matrix[j].count('P-') == level:
                best.append(j)
        if best:
            if "R" in matrix[best[0]]:
                G, prev, level = rec_rank(matrix, G, best[0], level, prev)

            else:
                if len(prev) == 0:
                    for node in best:
                        G.add_node(node, subset = level)
                else:
                    for node in best:
                        G.add_node(node, subset = level)
                        for previous in prev:
                            G.add_edge(previous, node)
                prev = best[:]
        level += 1
    return G
 
def combine_indifferent(partial_rank, labels):
    n = len(partial_rank)
    to_remove = set()
    for i in range(n):
        for j in range(i + 1, n):
            if partial_rank[i][j] == "I" and partial_rank[j][i] == "I":
                to_remove.add(j)
                labels[i] += f"\n{labels[j]}"
    
    new_partial_rank = [[partial_rank[i][j] for j in range(n) if j not in to_remove] for i in range(n) if i not in to_remove]
    new_labels = [labels[i] for i in range(n) if i not in to_remove]

    return new_partial_rank, new_labels

def draw_graph(matrix, labels,output_path:str):
    matrix, new_labels = combine_indifferent(matrix, labels)
    G = create_ranking(matrix)
    mapping = {i:new_labels[i] for i in range(len(new_labels))}
    G = nx.relabel_nodes(G, mapping)
    pos = nx.multipartite_layout(G, subset_key='subset')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=7650, font_size=10, font_weight='bold', arrowsize=50)
    plt.savefig(output_path)
