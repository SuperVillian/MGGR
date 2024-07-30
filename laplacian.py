import util
import numpy as np
import numpy.linalg as LA
from scipy.sparse.linalg import eigs
from numpy.linalg import eig

def sym_normalize_adj(adj):
    """
    Symmetrically normalize the adjacency matrix.
    
    Args:
        adj (np.ndarray): The adjacency matrix.

    Returns:
        np.ndarray: The symmetrically normalized adjacency matrix.
    """
    deg = adj.sum(1)
    deg_inv = np.where(deg > 0, 1. / np.sqrt(deg), 0)
    return np.einsum('i,ij,j->ij', deg_inv, adj, deg_inv)

def GCN_GC(adj, Q):
    """
    Perform graph coarsening using Graph Convolutional Networks (GCN).
    
    Args:
        adj (np.ndarray): The adjacency matrix.
        Q (np.ndarray): The coarsening matrix.

    Returns:
        np.ndarray: The coarsened graph.
    """
    L = sym_normalize_adj(adj)
    return util.lift_Q(Q).T @ L @ Q

def normalizeLaplacian(G):
    """
    Compute the normalized Laplacian of a graph.
    
    Args:
        G (np.ndarray): The adjacency matrix.

    Returns:
        np.ndarray: The normalized Laplacian matrix.
    """
    n = G.shape[0]
    return np.eye(n) - sym_normalize_adj(G)

def spectraLaplacian(G):
    """
    Compute the eigenvalues and eigenvectors of the normalized Laplacian.
    
    Args:
        G (np.ndarray): The adjacency matrix.

    Returns:
        tuple: The eigenvalues and eigenvectors.
    """
    L = normalizeLaplacian(G)
    L = (L + np.transpose(L)) / 2
    e, v = LA.eig(L)
    e = np.real(e)
    v = np.real(v)
    e_tmp = -e
    idx = e_tmp.argsort()[::-1]
    e = e[idx]
    v = v[:, idx]
    return e, v

def spectraLaplacian_two_end_n(G, n):
    """
    Compute the top-n and bottom-n eigenvalues and eigenvectors of the normalized Laplacian.
    
    Args:
        G (np.ndarray): The adjacency matrix.
        n (int): Number of eigenvalues and eigenvectors to compute.

    Returns:
        tuple: The top-n and bottom-n eigenvalues and eigenvectors.
    """
    N = G.shape[0]
    assert n <= N
    L = normalizeLaplacian(G)
    L = (L + np.transpose(L)) / 2
    e, v = eig(L)
    e = np.real(e)
    v = np.real(v)
    e_tmp = -e
    idx = e_tmp.argsort()[::-1]
    e = e[idx]
    v = v[:, idx]
    e1 = e[0:n]
    v1 = v[:, 0:n]
    e2 = e[N-n:N]
    v2 = v[:, N-n:N]
    return e1, v1, e2, v2

def spectraLaplacian_top_n(G, n):
    """
    Compute the top-n eigenvalues and eigenvectors of the normalized Laplacian.
    
    Args:
        G (np.ndarray): The adjacency matrix.
        n (int): Number of eigenvalues and eigenvectors to compute.

    Returns:
        tuple: The top-n eigenvalues and eigenvectors.
    """
    assert n <= G.shape[0]
    L = normalizeLaplacian(G)
    L = (L + np.transpose(L)) / 2
    e1, v1 = eig(L)
    e1 = np.real(e1)
    v1 = np.real(v1)
    e1_tmp = -e1
    idx = e1_tmp.argsort()[::-1]
    e1 = e1[idx]
    e1 = e1[0:n]
    v1 = v1[:, idx]
    v1 = v1[:, 0:n]
    return e1, v1
