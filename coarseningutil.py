import random, sys, functools
import numpy as np
import numpy.linalg as LA
import networkx as nx
from sklearn.cluster import KMeans
import coarsening

def multiply_Q(G, Q):
    """
    Multiply the graph matrix G with matrix Q for coarsening.
    
    Args:
        G (np.ndarray): The graph adjacency matrix.
        Q (np.ndarray): The matrix used for coarsening.

    Returns:
        np.ndarray: The coarsened graph matrix.
    """
    Gc = np.dot(np.dot(np.transpose(Q), G), Q)
    return Gc

def multiply_Q_lift(Gc, Q):
    """
    Lift the coarsened graph matrix Gc back to the original space using matrix Q.
    
    Args:
        Gc (np.ndarray): The coarsened graph matrix.
        Q (np.ndarray): The matrix used for lifting.

    Returns:
        np.ndarray: The lifted graph matrix.
    """
    G = np.dot(np.dot(Q, Gc), np.transpose(Gc))
    return G

def idx2Q(idx, n):
    """
    Convert index array to matrix Q.
    
    Args:
        idx (np.ndarray): Index array.
        n (int): Number of columns for matrix Q.

    Returns:
        np.ndarray: Matrix Q.
    """
    N = idx.shape[0]
    Q = np.zeros((N, n))
    for i in range(N):
        Q[i, idx[i]] = 1
    return Q

def Q2idx(Q):
    """
    Convert matrix Q to index array.
    
    Args:
        Q (np.ndarray): Matrix Q.

    Returns:
        np.ndarray: Index array.
    """
    N = Q.shape[0]
    n = Q.shape[1]
    idx = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for j in range(n):
            if Q[i, j] > 0:
                idx[i] = j
    return idx

def random_two_nodes(n):
    """
    Randomly select two nodes.
    
    Args:
        n (int): Number of nodes.

    Returns:
        tuple: Two randomly selected nodes.
    """
    perm = np.random.permutation(n)
    return perm[0], perm[1]

def merge_two_nodes(n, a, b):
    """
    Merge two nodes in the graph.
    
    Args:
        n (int): Number of nodes.
        a (int): Index of the first node.
        b (int): Index of the second node.

    Returns:
        np.ndarray: Matrix Q after merging.
    """
    assert a != b and a < n and b < n
    Q = np.zeros((n, n-1))
    cur = 0
    for i in range(n):
        if i == a or i == b:
            Q[i, n-2] = 1
        else:
            Q[i, cur] = 1
            cur = cur + 1
    return Q

def lift_Q(Q):
    """
    Lift matrix Q to another form.
    
    Args:
        Q (np.ndarray): Matrix Q.

    Returns:
        np.ndarray: Lifted matrix Q.
    """
    N = Q.shape[0]
    n = Q.shape[1]
    idx = np.zeros(N, dtype=np.int16)
    for i in range(N):
        for j in range(n):
            if Q[i, j] == 1:
                idx[i] = j
    d = np.zeros((n, 1))
    for i in range(N):
        d[idx[i]] = d[idx[i]] + 1

    Q2 = np.zeros((N, n))
    for i in range(N):
        Q2[i, idx[i]] = 1/d[idx[i]]
    return Q2

def ortho_Q(Q):
    """
    Orthogonalize matrix Q.
    
    Args:
        Q (np.ndarray): Matrix Q.

    Returns:
        np.ndarray: Orthogonalized matrix Q.
    """
    N = Q.shape[0]
    n = Q.shape[1]
    idx = np.zeros(N, dtype=np.int16)
    for i in range(N):
        for j in range(n):
            if Q[i, j] == 1:
                idx[i] = j
    d = np.zeros((n, 1))
    for i in range(N):
        d[idx[i]] = d[idx[i]] + 1

    Q2 = np.zeros((N, n))
    for i in range(N):
        Q2[i, idx[i]] = 1 / np.sqrt(d[idx[i]])
    return Q2

def orthoW_Q(Q, w):
    """
    Orthogonalize matrix Q with weights.
    
    Args:
        Q (np.ndarray): Matrix Q.
        w (np.ndarray): Weights array.

    Returns:
        np.ndarray: Orthogonalized matrix Q with weights.
    """
    N = Q.shape[0]
    n = Q.shape[1]
    if w is None:
        w = np.ones(N)
    c = Q.T @ w

    Q2 = (1/np.sqrt(c) * Q).T * w**0.5

    return Q2.T

def getReducedN(N, k, ratio):
    """
    Get the reduced number of nodes.
    
    Args:
        N (int): Original number of nodes.
        k (int): Minimum number of nodes.
        ratio (float): Coarsening ratio.

    Returns:
        int: Reduced number of nodes.
    """
    n = int(np.ceil(ratio*N))
    n = min(max([n, k]), N)

    return n

def get_h_init(Q, w=None):
    """
    Initialize h vector.
    
    Args:
        Q (np.ndarray): Matrix Q.
        w (np.ndarray): Weights array.

    Returns:
        np.ndarray: Initialized h vector.
    """
    N = Q.shape[0]
    if w is None: 
        w = np.ones(N) / N
    else:
        w = w / np.sum(w)

    nn = Q.T @ w
    h = w * (Q / nn).T

    return h

METHODS = ['mgc', 'sgc', 'wgc', 'vgc', 'vegc', 'sgwl', 'eig', 'none','GB_graph','GBC','GBGC']
OTHER_METHODS = ['vgc', 'vegc', 'mgc', 'sgc', 'sgwl', 'eig', 'none']

def assign_parser(parser):
    """
    Assign command-line arguments to the parser.
    
    Args:
        parser (argparse.ArgumentParser): Argument parser.

    """
    parser.add_argument('--dataset', type=str, default="MUTAG", help='name of dataset (default: MUTAG)')
    parser.add_argument('--method', type=str, default="wgc", help='name of the coarsening method')
    parser.add_argument('--ratio', type=float, default=0.2, help='the ratio between coarse and original graphs n/N')
    parser.add_argument('--NmaxRatio', action='store_true', help='in exp1, whether use the Nmax/logNmax ratio.')
    parser.add_argument('--runs', type=int, default=10, help='number of the runs, since coarsening method is random')
    parser.add_argument('--alm', type=str, default="", help='specify the agglomerative hierarchical clustering for wgc')
    parser.add_argument('--weighted', type=int, default=0, help='whether use weighted kernel kmeans in wgc')
    parser.add_argument('--normalized', type=int, default=0, help='whether use normalized laplacian in wgc')
    parser.add_argument('--cscale', action='store_true', help='in exp1, whether use the correct scale for other methods.')
    parser.add_argument('--ninit', type=int, default=10, help='n_init parameter for kmeans')
    parser.add_argument('--seed', type=int, default=42, help='the ratio between coarse and original graphs n/N')
    parser.add_argument('--save', type=int, default=0, help='if 1 then save the `res` variable')

def check_args(args):
    """
    Check validity of command-line arguments.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    if args.dataset not in ["MUTAG", "PROTEINS", "IMDB-BINARY", "tumblr_ct1", "DD", "COLLAB",
                            "MSRC_9", "PTC_MR", "AQSOL", "ZINC", "NCI1", "NCI109"]:
        print("Incorrect input dataset")
        sys.exit()
    if args.method not in METHODS:
        print(args.method, "Incorrect input coarsening method", METHODS)
        sys.exit()
    if args.ratio < 0 or args.ratio > 1:
        print("Incorrect input ratio")
        sys.exit()

def non_coarse(G, n):
    """
    Non-coarsening method that returns the original graph.
    
    Args:
        G (np.ndarray): Graph adjacency matrix.
        n (int): Number of nodes.

    Returns:
        tuple: Original graph matrix, identity matrix, and None.
    """
    return G, np.eye(G.shape[0]), None

def assign_method(args):
    """
    Assign the appropriate coarsening method based on command-line arguments.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        function: Coarsening method function.
    """
    if args.method == "vgc":
        coarse_method = functools.partial(coarsening.template_graph_coarsening, method="variation_neighborhoods")
    elif args.method == "vegc":
        coarse_method = functools.partial(coarsening.template_graph_coarsening, method="variation_edges")
    elif args.method == "mgc":
        coarse_method = coarsening.multilevel_graph_coarsening
    elif args.method == "sgc":
        coarse_method = coarsening.spectral_graph_coarsening
    elif args.method == "wgc":
        coarse_method = coarsening.weighted_graph_coarsening
    elif args.method == "GBC":
        coarse_method = coarsening.GBC
    elif args.method == 'sgwl':
        coarse_method = coarsening.SGWL
    elif args.method == "none":
        coarse_method = non_coarse
    elif args.method == "eig":
        coarse_method = non_coarse
    
    return coarse_method
