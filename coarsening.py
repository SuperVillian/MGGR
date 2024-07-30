import random, sys
import numpy as np
import numpy.linalg as LA
import networkx as nx
import coarseningutil
import measure
import laplacian
from sklearn.cluster import KMeans

def get_node_tag(graph):
    """
    Get the most common node tag in the graph.
    
    Args:
        graph (networkx.Graph): The input graph.

    Returns:
        int: The most common node tag.
    """
    node_labels = list(nx.get_node_attributes(graph, 'label').values())
    most_common_label = max(set(node_labels), key=node_labels.count)
    return most_common_label

def get_coarsening(graph, Q, n):
    """
    Perform coarsening on the graph using matrix Q.
    
    Args:
        graph (networkx.Graph): The input graph.
        Q (np.ndarray): Coarsening matrix.
        n (int): Number of clusters.

    Returns:
        tuple: The coarsened graph and node tags.
    """
    coarsening_idx = [[] for _ in range(n)]
    for i in range(len(Q)):
        for j in range(n):
            if Q[i][j] > 0:
                coarsening_idx[j].append(i)
    subgraphs = [nx.subgraph(graph, coarsening) for coarsening in coarsening_idx]
    if len(subgraphs) == 1:
        mapping = {old_node: new_node for new_node, old_node in enumerate(graph.nodes)}
        G_renumbered = nx.relabel_nodes(graph, mapping)
        node_tags = list(nx.get_node_attributes(graph, 'label').values())
        return G_renumbered, node_tags
    coarsening_node_tags = []
    coarsening_graph = nx.Graph()
    coarsening_graph.add_nodes_from([i for i in range(len(subgraphs))])
    for i in range(len(subgraphs)):
        for j in range(i + 1, len(subgraphs)):
            flag = False
            for a in subgraphs[i].nodes():
                for b in subgraphs[j].nodes():
                    if graph.has_edge(a, b):
                        flag = True
                        break
            if flag:
                coarsening_graph.add_edge(i, j)
        coarsening_node_tag = get_node_tag(subgraphs[i])
        coarsening_node_tags.append(coarsening_node_tag)
    return coarsening_graph, coarsening_node_tags

def multilevel_graph_coarsening(graph, ratio, node_tags):
    """
    Perform multilevel graph coarsening.
    
    Args:
        graph (networkx.Graph): The input graph.
        ratio (float): Coarsening ratio.
        node_tags (list): List of node tags.

    Returns:
        tuple: The coarsened graph and node tags.
    """
    labels = dict(zip(graph.nodes(), node_tags))
    nx.set_node_attributes(graph, labels, 'label')
    G = nx.to_numpy_array(graph)
    N = len(graph)
    n = int(np.ceil(ratio * N))
    Gc = G
    cur_size = N
    Q = np.eye(N)
    while cur_size > n:
        max_dist = float('inf')
        max_dist_a, max_dist_b = -1, -1
        for i in range(cur_size):
            for j in range(i + 1, cur_size):
                dist = measure.normalized_L1(Gc[i], Gc[j])
                if dist < max_dist:
                    max_dist = dist
                    max_dist_a, max_dist_b = i, j
        if max_dist_a == -1:
            max_dist_a, max_dist_b = coarseningutil.random_two_nodes(cur_size)
        cur_Q = coarseningutil.merge_two_nodes(cur_size, max_dist_a, max_dist_b)
        Q = np.dot(Q, cur_Q)
        Gc = coarseningutil.multiply_Q(Gc, cur_Q)
        cur_size = Gc.shape[0]
    coarsening_graph, coarsening_node_tags = get_coarsening(graph, Q, n)
    return coarsening_graph, coarsening_node_tags

def regular_partition(N, n):
    """
    Regular partition of N nodes into n clusters.
    
    Args:
        N (int): Number of nodes.
        n (int): Number of clusters.

    Returns:
        np.ndarray: Index array for partition.
    """
    block_size = N // n + 1 if N % n == 0 else N // (n - 1) + 1
    idx = np.zeros(N, dtype=np.int32)
    for i in range(N):
        idx[i] = i // block_size
    return idx

from graph_coarsening.coarsening_utils import *
import pygsp

def template_graph_coarsening_vgc(graph, ratio, node_tags):
    """
    Perform template graph coarsening using variation neighborhoods method.
    
    Args:
        graph (networkx.Graph): The input graph.
        ratio (float): Coarsening ratio.
        node_tags (list): List of node tags.

    Returns:
        tuple: The coarsened graph and node tags.
    """
    labels = dict(zip(graph.nodes(), node_tags))
    nx.set_node_attributes(graph, labels, 'label')
    G = nx.to_numpy_array(graph)
    G = pygsp.graphs.Graph(G)
    N = len(graph)
    n = int(np.ceil(ratio * N))
    C, _, _, _ = coarsen(G, r=n, method="variation_neighborhoods")
    n = C.shape[0]
    Q = np.zeros((N, n), dtype=np.int32)
    Q[C.toarray().T > 0] = 1
    coarsening_graph, coarsening_node_tags = get_coarsening(graph, Q, n)
    return coarsening_graph, coarsening_node_tags

def template_graph_coarsening_vegc(graph, ratio, node_tags):
    """
    Perform template graph coarsening using variation edges method.
    
    Args:
        graph (networkx.Graph): The input graph.
        ratio (float): Coarsening ratio.
        node_tags (list): List of node tags.

    Returns:
        tuple: The coarsened graph and node tags.
    """
    labels = dict(zip(graph.nodes(), node_tags))
    nx.set_node_attributes(graph, labels, 'label')
    G = nx.to_numpy_array(graph)
    G = pygsp.graphs.Graph(G)
    N = len(graph)
    n = int(np.ceil(ratio * N))
    C, _, _, _ = coarsen(G, r=n, method="variation_edges")
    n = C.shape[0]
    Q = np.zeros((N, n), dtype=np.int32)
    Q[C.toarray().T > 0] = 1
    coarsening_graph, coarsening_node_tags = get_coarsening(graph, Q, n)
    return coarsening_graph, coarsening_node_tags

def variation_nbhd_graph_coarsening(am, n, **kwargs):
    """
    Perform variation neighborhoods graph coarsening.
    
    Args:
        am (np.ndarray): Adjacency matrix.
        n (int): Number of clusters.

    Returns:
        tuple: Coarsened graph matrix, coarsening matrix Q, and index array.
    """
    G = pygsp.graphs.Graph(am)
    N = G.N
    C, _, _, _ = coarsen(G, r=n, method="variation_neighborhood")
    n = C.shape[0]
    Q = np.zeros((N, n), dtype=np.int32)
    Q[C.toarray().T > 0] = 1
    idx = coarseningutil.Q2idx(Q)
    return coarseningutil.multiply_Q(am, Q), Q, idx

def variation_edge_graph_coarsening(am, n, **kwargs):
    """
    Perform variation edges graph coarsening.
    
    Args:
        am (np.ndarray): Adjacency matrix.
        n (int): Number of clusters.

    Returns:
        tuple: Coarsened graph matrix, coarsening matrix Q, and index array.
    """
    G = pygsp.graphs.Graph(am)
    N = G.N
    C, _, _, _ = coarsen(G, r=n, method="variation_edges")
    n = C.shape[0]
    Q = np.zeros((N, n), dtype=np.int32)
    Q[C.toarray().T > 0] = 1
    idx = coarseningutil.Q2idx(Q)
    return coarseningutil.multiply_Q(am, Q), Q, idx

def spectral_graph_coarsening(graph, ratio, node_tags):
    """
    Perform spectral graph coarsening.
    
    Args:
        graph (networkx.Graph): The input graph.
        ratio (float): Coarsening ratio.
        node_tags (list): List of node tags.

    Returns:
        tuple: The coarsened graph and node tags.
    """
    labels = dict(zip(graph.nodes(), node_tags))
    nx.set_node_attributes(graph, labels, 'label')
    G = nx.to_numpy_array(graph)
    N = len(graph)
    n = int(np.ceil(ratio * N))
    e1, v1, e2, v2 = laplacian.spectraLaplacian_two_end_n(G, n)
    min_dist = n + 1
    if n >= N:
        Q = np.eye(N)
        return G, Q, coarseningutil.Q2idx(Q)
    for k in range(0, n):
        if e1[k] <= 1:
            if k + 1 < n and e

2[k + 1] < 1:
                continue
            if k + 1 < n and e2[k + 1] >= 1:
                v_all = np.concatenate((v1[:, 0:(k + 1)], v2[:, (k + 1):n]), axis=1)
            elif k == n - 1:
                v_all = v1[:, 0:n]
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10).fit(v_all)
            idx = kmeans.labels_
            sumd = kmeans.inertia_
            Q = coarseningutil.idx2Q(idx, n)
            Gc = coarseningutil.multiply_Q(G, Q)
            ec, vc = laplacian.spectraLaplacian(Gc)
            dist = measure.eig_partial_dist_k_two_end_n(e1, e2, ec, k)
            if dist < min_dist:
                min_dist = dist
                idx_min = idx
                min_sumd = sumd
                Gc_min = Gc
                Q_min = Q
    coarsening_graph, coarsening_node_tags = get_coarsening(graph, Q_min, n)
    return coarsening_graph, coarsening_node_tags

from pygkernels.cluster import KKMeans, KKMeans_iterative
from ot.backend import get_backend

def weighted_graph_coarsening(graph, ratio, node_tags, scale=0, seed=42, n_init=10, 
                              sample_weight=None, init='k-means++', h_init=None, tol_empty=False):
    """
    Perform weighted graph coarsening.
    
    Args:
        graph (networkx.Graph): The input graph.
        ratio (float): Coarsening ratio.
        node_tags (list): List of node tags.
        scale (int): Scaling option.
        seed (int): Random seed.
        n_init (int): Number of initializations for KMeans.
        sample_weight (np.ndarray): Sample weights.
        init (str): Initialization method for KMeans.
        h_init (np.ndarray): Initial guess for KMeans.
        tol_empty (bool): Tolerance for empty clusters.

    Returns:
        tuple: The coarsened graph and node tags.
    """
    labels = dict(zip(graph.nodes(), node_tags))
    nx.set_node_attributes(graph, labels, 'label')
    G = nx.to_numpy_array(graph)
    bG = get_backend(G)
    S = bG.diag(bG.sum(G, axis=0)) + G
    N = len(graph)
    n = int(np.ceil(ratio * N))
    kmeans = KKMeans(n_clusters=n, n_init=n_init, init=init, 
                     init_measure='inertia', random_state=seed)
    idx = kmeans.predict(S, A=h_init, sample_weight=sample_weight, tol_empty=tol_empty)
    if idx is None:
        kmeans = KKMeans(n_clusters=n, n_init=n_init, init="k-means++", 
                         init_measure='inertia', random_state=seed)
        idx = kmeans.predict(S, A=None, sample_weight=sample_weight, tol_empty=tol_empty)
    Q = coarseningutil.idx2Q(idx, n)
    if scale == 0: 
        Q2 = coarseningutil.lift_Q(Q)
    elif scale == 1: 
        Q2 = coarseningutil.orthoW_Q(Q, sample_weight)
    else:
        Q2 = Q
    Gc = coarseningutil.multiply_Q(S, Q2)
    coarsening_graph, coarsening_node_tags = get_coarsening(graph, Q2, n)
    return coarsening_graph, coarsening_node_tags

def spectral_clustering(G, n):
    """
    Perform spectral clustering.
    
    Args:
        G (np.ndarray): Graph adjacency matrix.
        n (int): Number of clusters.

    Returns:
        tuple: Coarsened graph matrix, coarsening matrix Q, and index array.
    """
    N = G.shape[0]
    e1, v1 = laplacian.spectraLaplacian_top_n(G, n)
    v_all = v1[:, 0:n]
    kmeans = KMeans(n_clusters=n).fit(v_all)
    idx = kmeans.labels_
    sumd = kmeans.inertia_
    Q = coarseningutil.idx2Q(idx, n)
    Gc = coarseningutil.multiply_Q(G, Q)
    return Gc, Q, idx

def get_random_partition(N, n):
    """
    Get a random partition of N nodes into n clusters.
    
    Args:
        N (int): Number of nodes.
        n (int): Number of clusters.

    Returns:
        np.ndarray: Index array for partition.
    """
    for _ in range(500):
        flag = True
        a = np.zeros(N, dtype=np.int64)
        cnt = np.zeros(n, dtype=np.int64)
        for j in range(N):
            a[j] = random.randint(0, n - 1)
            cnt[a[j]] += 1
        if all(cnt):
            return a
    return a
