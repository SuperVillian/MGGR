import networkx as nx
import numpy as np
import random
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import sys

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 4, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=4, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list



def process_glist(g_list, label_dict, degree_as_tag):
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list) 

        g.label = label_dict[g.label] 

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1) 
        
     
    if degree_as_tag: 
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
    

    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags)) 
    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}


    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

from GBGC_addall_feature import GB_coarsen

def GB_load_data(dataset, degree_as_tag, purity, degree_purity):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('GB_load_data')
    g_list = []
    graph_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset_our/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip()) 

        count = 0 

        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row] 
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped 

            g = nx.Graph()
            node_tags = []
            n_edges = 0
            for j in range(n):
                g.add_node(j) 

                row = f.readline().strip().split()
                tmp = int(row[1]) + 2 

                if tmp == len(row): 

                    row = [int(w) for w in row]
                    attr = None
                    node_feature_flag = False
                else:
                    row, attr = [int(w) for w in row[:tmp]], torch.tensor([float(w) for w in row[tmp:]])
                    node_feature_flag = True
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped 
                node_tags.append(feat_dict[row[0]])


                if node_feature_flag == True:
                    nx.set_node_attributes(g,{j: {'attributes': attr}})
                    

                n_edges += row[1]
                for k in range(2, len(row)): 
                    g.add_edge(j, row[k])
            graph_list.append((g, node_tags, l))

        if node_feature_flag == False:
            add_feat(graph_list)
        
        
    i = 1
    for item in graph_list:
        g = item[0]
        dim = len(nx.get_node_attributes(g, 'attributes')[0])
        node_tags = item[1]
        l = item[2]
        if len(g)<5:
            GB_g, GB_node_tags = g, node_tags
            count += 1
 
        else:
            GB_g, GB_node_tags = GB_coarsen(g, node_tags, purity, degree_purity)

        if len(g) * 0.1 > len(GB_g) or len(GB_g) <=2:
            GB_g, GB_node_tags = g, node_tags
            count += 1

        i += 1
        expand_dim1(GB_g, dim+10)
        g_list.append(S2VGraph(GB_g, l, GB_node_tags)) 



    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list) 

        g.label = label_dict[g.label] 


        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1) 
        

    if degree_as_tag: 
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
    

    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags)) 
    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    k = 0
    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset)+len(g.g.nodes[0]["attributes"]))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        i = 0
        for node in g.g.nodes():
            for j in range(len(tagset), len(tagset)+len(g.g.nodes[0]["attributes"])):
                g.node_features[i, j] = g.g.nodes[node]["attributes"][j-len(tagset)]
            i = i + 1
    all_features = torch.cat([g.node_features for g in g_list], dim=0)

    normalized_features = F.normalize(all_features, p=2, dim=0)

    start_idx = 0
    for g in g_list:
        end_idx = start_idx + g.node_features.shape[0]
        g.node_features = normalized_features[start_idx:end_idx]

        start_idx = end_idx

    return g_list, len(label_dict)

def add_feat(graph_list):
    tagset = set([])
    for item in graph_list:
        tagset = tagset.union(set(item[1]))
    tagset = list(tagset)

    for item in graph_list:
        g = item[0]
        node_tags = item[1]
        for node in g.nodes():  
            node_feature = torch.zeros(len(tagset))
            node_feature[node_tags[node]] = 1
            nx.set_node_attributes(g,{node: {'attributes': node_feature}})


def expand_dim1(graph, dim):
     for node_id, node_data in graph.nodes(data=True):
        attrs = node_data['attributes']
        current_size = attrs.size(-1)  
        if current_size < dim:
            padding_size = dim - current_size 
            padded_attrs = F.pad(attrs, (0, padding_size), 'constant', 0)
            nx.set_node_attributes(graph, {node_id: {'attributes': padded_attrs}})
    



        


