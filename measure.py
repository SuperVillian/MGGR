import random
import numpy as np
import numpy.linalg as LA

def NMI(e1, e2, k):
    """
    Compute the Normalized Mutual Information (NMI) between two sets of labels.
    
    Args:
        e1 (np.ndarray): First set of labels.
        e2 (np.ndarray): Second set of labels.
        k (int): Number of clusters.

    Returns:
        float: The NMI value.
    """
    cnt_e1 = np.zeros(k)
    cnt_e2 = np.zeros(k)
    n = e1.shape[0]
    for i in range(k):
        cnt_e1[i] = np.sum(e1 == i)
        cnt_e2[i] = np.sum(e2 == i)

    p_e1 = cnt_e1 / n
    log_e1 = np.zeros(k)
    for i in range(k):
        if cnt_e1[i] != 0:
            log_e1[i] = np.log(p_e1[i])
    e_e1 = - np.dot(p_e1, log_e1)

    p_e2 = cnt_e2 / n
    log_e2 = np.zeros(k)
    for i in range(k):
        if cnt_e2[i] != 0:
            log_e2[i] = np.log(p_e2[i])
    e_e2 = - np.dot(p_e2, log_e2)

    MI = 0
    for i in range(k):
        for j in range(k):
            idx_e1 = (e1 == i)
            idx_e2 = (e2 == j)
            idx_overlap = np.sum(idx_e1 * idx_e2)
            if idx_overlap == 0:
                continue
            idx_p = idx_overlap / n
            MI += idx_p * np.log(idx_p / (p_e1[i] * p_e2[j]))

    result = MI / ((e_e1 + e_e2) / 2)
    return result

def cosine_distance(e1, e2):
    """
    Compute the cosine distance between two vectors.
    
    Args:
        e1 (np.ndarray): First vector.
        e2 (np.ndarray): Second vector.

    Returns:
        float: The cosine distance.
    """
    s1 = np.sum(e1)
    s2 = np.sum(e2)
    if s1 == 0 or s2 == 0:
        return 0
    return np.dot(e1, e2) / (LA.norm(e1) * LA.norm(e2))

def normalized_L1(e1, e2):
    """
    Compute the normalized L1 distance between two vectors.
    
    Args:
        e1 (np.ndarray): First vector.
        e2 (np.ndarray): Second vector.

    Returns:
        float: The normalized L1 distance.
    """
    s1 = np.sum(e1)
    norm_e1 = e1 / s1 if s1 != 0 else np.zeros(e1.size)
    s2 = np.sum(e2)
    norm_e2 = e2 / s2 if s2 != 0 else np.zeros(e2.size)
    return np.sum(abs(norm_e1 - norm_e2))

def heavy_edge_weight(e, d1, d2):
    """
    Compute the heavy edge weight between two nodes.
    
    Args:
        e (float): Edge weight.
        d1 (float): Degree of the first node.
        d2 (float): Degree of the second node.

    Returns:
        float: The heavy edge weight.
    """
    if d1 < 0.00001 or d2 < 0.00001:
        return 0
    return e / max(d1, d2)

def eig_full_dist(e1, e2):
    """
    Compute the full eigenvalue distance between two sets of eigenvalues.
    
    Args:
        e1 (np.ndarray): First set of eigenvalues.
        e2 (np.ndarray): Second set of eigenvalues.

    Returns:
        float: The full eigenvalue distance.
    """
    dist = 0
    N = e1.shape[0]
    n = e2.shape[0]
    e2 = np.concatenate((e2, np.ones(N - n)), axis=None)
    e2 = np.sort(e2, axis=None)
    for i in range(N):
        dist += np.abs(e1[i] - e2[i])
    return dist

def eig_partial_dist_k_two_end_n(e1, e2, ec, k):
    """
    Compute the partial eigenvalue distance for the top and bottom k eigenvalues.
    
    Args:
        e1 (np.ndarray): First set of eigenvalues.
        e2 (np.ndarray): Second set of eigenvalues.
        ec (np.ndarray): Coarsened set of eigenvalues.
        k (int): Number of eigenvalues to consider.

    Returns:
        float: The partial eigenvalue distance.
    """
    assert e1.shape[0] == e2.shape[0] and e1.shape[0] == ec.shape[0]
    n = ec.shape[0]
    dist = 0
    for i in range(0, k + 1):
        assert ec[i] - e1[i] > -1e-8
        dist += np.abs(ec[i] - e1[i])
    for i in range(k + 1, n):
        assert e2[i] - ec[i] > -1e-8
        dist += np.abs(e2[i] - ec[i])
    return dist

def eig_partial_dist_k(e, ec, k):
    """
    Compute the partial eigenvalue distance for the top k eigenvalues.
    
    Args:
        e (np.ndarray): First set of eigenvalues.
        ec (np.ndarray): Coarsened set of eigenvalues.
        k (int): Number of eigenvalues to consider.

    Returns:
        float: The partial eigenvalue distance.
    """
    N = e.shape[0]
    n = ec.shape[0]
    dist = 0
    for i in range(0, k):
        assert ec[i] - e[i] > -1e-8
        dist += np.abs(ec[i] - e[i])
    for i in range(k, n):
        assert e[N - n + i] - ec[i] > -1e-8
        dist += np.abs(ec[i] - e[N - n + i])
    return dist

def eig_partial_dist(e, ec):
    """
    Compute the minimum partial eigenvalue distance.
    
    Args:
        e (np.ndarray): First set of eigenvalues.
        ec (np.ndarray): Coarsened set of eigenvalues.

    Returns:
        tuple: The minimum distance and the corresponding k value.
    """
    N = e.shape[0]
    n = ec.shape[0]
    min_dist = N + 1
    min_k = -1
    for i in range(1, n + 1):
        dist = eig_partial_dist_k(e, ec, i)
        if dist < min_dist:
            min_dist = dist
            min_k = i
    return min_dist, min_k
