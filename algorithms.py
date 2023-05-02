# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph

# Library for INT_MAX
import math
import sys
import time
from collections import defaultdict
import random

import numpy as np

import networkx as nx
from memory_profiler import memory_usage


def printSolution(dist):
    print("Vertex \tDistance from Source")
    for node in range(dist.shape[0]):
        print(node, "\t", dist[node])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree


def minDistance(dist, sptSet):
    # Initialize minimum distance for next node
    min = sys.maxsize

    # Search not nearest vertex not in the
    # shortest path tree
    for u in range(dist.shape[0]):
        if dist[u] < min and not sptSet[u]:
            min = dist[u]
            min_index = u

    return min_index

    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation


# Function to add an edge to graph
def toDictList(graph):
    # graph: Tuple containing two N length arrays containing src and dest indices
    dictlist = defaultdict(list)
    for i, src in enumerate(graph[0]):
        dictlist[src].append(graph[1][i])
    return dictlist
    # graph[u].append(v)


# A function used by DFS

def DFSUtil(graph, v, visited):
    # Mark the current node as visited and print it
    visited.add(v)
    print(v, end=" ")

    # recur for all the vertices adjacent to this vertex
    for neighbour in graph[v]:
        if neighbour not in visited:
            DFSUtil(graph, neighbour, visited)


# The function to do DFS traversal. It uses recursive DFSUtil

def DFS(graph):
    # create a set to store all visited vertices
    visited = set()
    # call the recursive helper function to print DFS traversal starting from all
    # vertices one by one
    for vertex in graph:
        if vertex not in visited:
            DFSUtil(graph, vertex, visited)


def dijkstra(graph, src, do_print=True):
    V = graph.shape[0]  # number of nodes from adjacency matrix
    dist = np.full((V,), sys.maxsize)  # [sys.maxsize] * V
    dist[src] = 0
    sptSet = [False] * V

    for cout in range(V):

        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # x is always equal to src in first iteration
        x = minDistance(dist, sptSet)

        # Put the minimum distance vertex in the
        # shortest path tree
        sptSet[x] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shortest path tree
        for y in range(V):
            if graph[x, y] > 0 and not sptSet[y] and \
                    dist[y] > dist[x] + graph[x, y]:
                dist[y] = dist[x] + graph[x, y]
    if do_print:
        printSolution(dist)


def experiment(algorithm, args):
    tick = time.time()

    max_memory_used = sum(memory_usage((algorithm, args)))

    tock = time.time()
    time_elapsed = tock - tick

    return float(f'{time_elapsed:.2f}'), float(f'{max_memory_used:.3f}')


if __name__ == "__main__":
    """
    graph_list = np.nonzero(sample_graph)"""
    print('####################################################################)')
    print('# Dijkstra vs DFS #')
    print('# Time (s), Total Memory (MiB) #\n')

    print()

    # graph datasets #

    geeks_graph = np.array([[0, 4, 0, 0, 0, 0, 0, 8, 0],
                            [4, 0, 8, 0, 0, 0, 0, 11, 0],
                            [0, 8, 0, 7, 0, 4, 0, 0, 2],
                            [0, 0, 7, 0, 9, 14, 0, 0, 0],
                            [0, 0, 0, 9, 0, 10, 0, 0, 0],
                            [0, 0, 4, 14, 10, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 2, 0, 1, 6],
                            [8, 11, 0, 0, 0, 0, 1, 0, 7],
                            [0, 0, 2, 0, 0, 0, 6, 7, 0]])
    geeks_graph_list = toDictList(np.nonzero(geeks_graph))
    geeks_graph = nx.from_numpy_array(geeks_graph)

    karate_club_graph = nx.karate_club_graph()
    karate_club_graph_list = toDictList(np.nonzero(nx.to_numpy_array(karate_club_graph)))

    n = 10  # of nodes
    m = 20  # of edges
    seed = 20160  # seed random number generators for reproducibility
    # Use seed for reproducibility
    random_graph_1 = nx.gnm_random_graph(n, m, seed=seed)
    random_graph_1_list = toDictList(np.nonzero(nx.to_numpy_array(random_graph_1)))

    n = 100  # of nodes
    m = 200  # of edges
    random_graph_2 = nx.gnm_random_graph(n, m, seed=seed)
    random_graph_2_list = toDictList(np.nonzero(nx.to_numpy_array(random_graph_2)))

    n = 1000  # of nodes
    m = 2000  # of edges
    random_graph_3 = nx.gnm_random_graph(n, m, seed=seed)
    random_graph_3_list = toDictList(np.nonzero(nx.to_numpy_array(random_graph_3)))

    n = 10000  # of nodes
    m = 20000  # of edges
    random_graph_4 = nx.gnm_random_graph(n, m, seed=seed)
    random_graph_4_list = toDictList(np.nonzero(nx.to_numpy_array(random_graph_4)))

    # make a random graph of N nodes with expected degrees of E
    n = 1000  # n nodes
    p = 0.1
    w = [p * n for i in range(n)]  # w = p*n for all nodes
    conf_model_graph = nx.expected_degree_graph(w)  # configuration model
    conf_model_graph_1_list = toDictList(np.nonzero(nx.to_numpy_array(conf_model_graph)))

    # tests #

    # test 1 - geeks
    sample = random.sample(range(geeks_graph.number_of_nodes()), 2)
    print('\nGeeks graph')
    print('From source node: ' + str(sample[0]))
    print('Dijkstra: ', experiment(nx.single_source_dijkstra_path,
                                   (geeks_graph,) + tuple(sample)))
    print('DFS: ', experiment(nx.dfs_predecessors,(geeks_graph, sample[0])))

    # test 2 - karate club
    sample = random.sample(range(karate_club_graph.number_of_nodes()), 2)
    print('\nKarate club graph')
    print('From source node: ' + str(sample[0]))
    print('Dijkstra: ', experiment(nx.single_source_dijkstra_path,
                                   (karate_club_graph,) + tuple(sample)))
    print('DFS: ', experiment(nx.dfs_predecessors,
                              (karate_club_graph, sample[0])))

    # test 3
    # erdos renyi n=10
    sample = random.sample(range(random_graph_1.number_of_nodes()), 2)
    print('\nErdos renyi graph |N|=10, |E|=20')
    print('From source node: ' + str(sample[0]))
    print('Dijkstra: ', experiment(nx.single_source_dijkstra_path,
                                   (random_graph_1,) + tuple(sample)))
    print('DFS: ', experiment(nx.dfs_predecessors,
                              (random_graph_1, sample[0])))

    # erdos renyi n=100
    sample = random.sample(range(random_graph_2.number_of_nodes()), 2)
    print('\nErdos renyi graph |N|=100, |E|=200')
    print('From source node: ' + str(sample[0]))
    print('Dijkstra: ', experiment(nx.single_source_dijkstra_path,
                                   (random_graph_2,) + tuple(sample)))
    print('DFS: ', experiment(nx.dfs_predecessors,
                              (random_graph_2, sample[0])))

    # erdos renyi n=1000
    sample = random.sample(range(random_graph_3.number_of_nodes()), 2)
    print('\nErdos renyi graph |N|=1000, |E|=2000')
    print('From source node: ' + str(sample[0]))
    print('Dijkstra: ', experiment(nx.single_source_dijkstra_path,
                                   (random_graph_3,) + tuple(sample)))
    print('DFS: ', experiment(nx.dfs_predecessors,
                              (random_graph_3, sample[0])))

    # erdos renyi n=10,000
    sample = random.sample(range(random_graph_4.number_of_nodes()), 2)
    print('\nErdos renyi graph |N|=10,000, |E|=20,000')
    print('From source node: ' + str(sample[0]))
    print('Dijkstra: ', experiment(nx.single_source_dijkstra_path,
                                   (random_graph_4,) + tuple(sample)))
    print('DFS: ', experiment(nx.dfs_predecessors,
                              (random_graph_4, sample[0])))

    # test 4 - configuration model
    sample = random.sample(range(conf_model_graph.number_of_nodes()), 2)
    print('\nConfiguration model graph |N|=10,000, p=0.1')
    print('From source node: ' + str(sample[0]))
    print('Dijkstra: ', experiment(nx.single_source_dijkstra_path,
                                   (conf_model_graph,) + tuple(sample)))
    print('DFS: ', experiment(nx.dfs_predecessors,
                              (conf_model_graph, sample[0])))

