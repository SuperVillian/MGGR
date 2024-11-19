import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import math
import sys

def init_GB_graph(graph, init_GB_num):
    degree_dict = dict(graph.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)

    center_nodes = [sorted_nodes[0]]  
    center_paths = []
    center_paths.append(nx.single_source_shortest_path_length(graph, source=sorted_nodes[0]))
    for _ in range(1, init_GB_num):
        to_center_list = []
        for node in sorted_nodes:
            if node not in center_nodes:
                min_to_center = float('inf')
                for i in range(len(center_paths)):
                    now_distance = center_paths[i][node]
                    if now_distance < min_to_center:
                        min_to_center = now_distance
                to_center_list.append([node, min_to_center])
        next_node = max(to_center_list, key=lambda x:x[1])
        center_nodes.append(next_node[0])
        center_paths.append(nx.single_source_shortest_path_length(graph, source=next_node[0]))

    point_nodes = [node for node in sorted_nodes if node not in center_nodes]

    center_paths = []
    clusters = []
    for center in center_nodes:
        center_paths.append(nx.single_source_shortest_path_length(graph, source=center))
        clusters.append([center])

    for point in point_nodes:
        point_to_center_len = float('inf')
        point_to_center_idx = 0
        for idx, center_path in enumerate(center_paths):
            if point_to_center_len > center_path.get(point, float('inf')):
                point_to_center_len = center_path[point]
                point_to_center_idx = idx
        clusters[point_to_center_idx].append(point)

    init_GB_list = [nx.subgraph(graph, cluster) for cluster in clusters]
    return init_GB_list



def Calculate_qity(graph):

    avg_degree = graph.number_of_edges() / len(graph)
    
    betweenness_centrality = nx.betweenness_centrality(graph)
    max_betweenness = max(betweenness_centrality.values())

    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    modularity = nx.algorithms.community.modularity(graph, communities)

    qity = avg_degree * (1 - max_betweenness) + 0.5 * modularity
    
    return qity



def Calculate_purity(graph):
    node_labels = list(nx.get_node_attributes(graph, 'label').values())

    most_common_label = max(set(node_labels), key=node_labels.count)

    purity = node_labels.count(most_common_label) / len(node_labels)

    return purity


def Calculate_qity_purity(graph):
    purity = Calculate_purity(graph)
    avg_degree = graph.number_of_edges() / len(graph)
    if purity == 1:
        return avg_degree
    qity = avg_degree / (1 - purity)
    return qity


def split_ball_split(graph):
    degree_dict = dict(graph.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
    center_nodes = sorted_nodes[:2]
    point_nodes = sorted_nodes[2:]
    
    center_paths = []
    clusters = []
    for center in center_nodes:
        center_paths.append(nx.single_source_shortest_path_length(graph, source=center))
        clusters.append([center])
    for point in point_nodes:
        point_to_center_len = float('inf')
        idx = 0
        for center_path in center_paths:
            if point_to_center_len > center_path[point]:
                point_to_center_len = center_path[point]

                point_to_center_idx = idx
            idx += 1
        clusters[point_to_center_idx].append(point)
    cluster_a = clusters[0]  
    cluster_b = clusters[1]           
    
    graph_a = nx.subgraph(graph, cluster_a)
    graph_b = nx.subgraph(graph, cluster_b)
    return graph_a, graph_b

def split_ball(graph, split_GB_list, purity):
    if len(graph) == 1:
        split_GB_list.append(graph)
        return 

    if purity in [0, 2, 3]:
        graph_a, graph_b = split_ball_split(graph)
        if len(graph_a.edges()) == 0 or len(graph_b.edges()) == 0:
            split_GB_list.append(graph)
        else:
            if purity == 0:
                qity = Calculate_qity(graph)
                qity_a = Calculate_qity(graph_a)
                qity_b = Calculate_qity(graph_b)
            elif purity == 2:
                qity = Calculate_purity(graph)
                qity_a = Calculate_purity(graph_a)
                qity_b = Calculate_purity(graph_b)
            else:
                qity = Calculate_qity_purity(graph)
                qity_a = Calculate_qity_purity(graph_a)
                qity_b = Calculate_qity_purity(graph_b)

            if qity < qity_a + qity_b:
                split_ball(graph_a, split_GB_list, purity)
                split_ball(graph_b, split_GB_list, purity)
            else:
                split_GB_list.append(graph)
    else:
        qity = Calculate_purity(graph)
        if qity < purity:
            graph_a, graph_b = split_ball_split(graph)
            split_ball(graph_a, split_GB_list, purity)
            split_ball(graph_b, split_GB_list, purity)
        else:
            split_GB_list.append(graph)

def split_ball_purity(graph, split_GB_list, purity):
    if len(graph) == 1:
        split_GB_list.append(graph)
        return 

    if purity == 2:
        graph_a, graph_b = split_ball_split(graph)
        if len(graph_a.edges()) == 0 or len(graph_b.edges()) == 0:
            split_GB_list.append(graph)
        else:
            qity = Calculate_purity(graph)
            qity_a = Calculate_purity(graph_a)
            qity_b = Calculate_purity(graph_b)
            if qity < qity_a + qity_b:
                split_ball(graph_a, split_GB_list, purity)
                split_ball(graph_b, split_GB_list, purity)
            else:
                split_GB_list.append(graph)
    else:
        qity = Calculate_purity(graph)
        if qity < purity:
            graph_a, graph_b = split_ball_split(graph)
            split_ball(graph_a, split_GB_list, purity)
            split_ball(graph_b, split_GB_list, purity)
        else:
            split_GB_list.append(graph)

def get_node_tag(graph):
    node_labels = list(nx.get_node_attributes(graph, 'label').values())

    most_common_label = max(set(node_labels), key=node_labels.count)

    return most_common_label

def get_GB_graph(graph, purity, degree_purity):
    if len(graph) <= 2:
        mapping = {old_node: new_node for new_node, old_node in enumerate(graph.nodes)}
        G_renumbered = nx.relabel_nodes(graph, mapping)
        node_tags = list(nx.get_node_attributes(graph, 'label').values())
        return G_renumbered, node_tags

    init_GB_num = math.isqrt(len(graph))
    init_GB_list = init_GB_graph(graph, init_GB_num)
    GB_list = []
    if degree_purity == False:
        for init_GB in init_GB_list:
            split_GB_list = []
            split_ball(init_GB, split_GB_list, purity)
            GB_list.extend(split_GB_list)
    else:
        for init_GB in init_GB_list:
            split_GB_list = []
            split_ball(init_GB, split_GB_list, 0)
            GB_list.extend(split_GB_list)
        purity_GB_list = []
        for GB in GB_list:
            split_GB_list = []
            split_ball_purity(GB, split_GB_list, purity)
            purity_GB_list.extend(split_GB_list)
        
        GB_list = purity_GB_list


    if len(GB_list) == 1:
        mapping = {old_node: new_node for new_node, old_node in enumerate(graph.nodes)}
        G_renumbered = nx.relabel_nodes(graph, mapping)
        node_tags = list(nx.get_node_attributes(graph, 'label').values())
        return G_renumbered, node_tags

    GB_node_tags = []
    GB_graph = nx.Graph()
    GB_graph.add_nodes_from([i for i in range(len(GB_list))])
    for index in range(len(GB_list)):
        GB_agg_attr = torch.stack([sum(values) for values in zip(*list(nx.get_node_attributes(GB_list[index], 'attributes').values()))])
        GB_aft_attr = add_GB_aft_attr(GB_list[index])
        GB_attr = torch.cat((GB_agg_attr, GB_aft_attr), dim=0)
        nx.set_node_attributes(GB_graph,{index:{'attributes': GB_attr}})

    for i in range(len(GB_list)):
        for j in range(i+1, len(GB_list)):
            flag = False
            count = 0
            for a in GB_list[i].nodes():
                for b in GB_list[j].nodes():
                    if graph.has_edge(a, b):
                        count += 1
                        flag = True
            if flag:
                GB_graph.add_edge(i, j)
        GB_node_tag = get_node_tag(GB_list[i])
        nx.set_node_attributes(GB_graph,{i:{'label':GB_node_tag}})
        GB_node_tags.append(GB_node_tag)
    return GB_graph, GB_node_tags

    
def add_GB_aft_attr(graph):
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        if graph.size() == 0:
            edge_betweenness_centrality = 0
        else:
            edge_betweenness_centrality = sum(dict(nx.edge_betweenness_centrality(graph)).values()) / num_nodes
        triangles = sum(dict(nx.triangles(graph)).values()) / num_nodes
        closeness_centrality = sum(dict(nx.closeness_centrality(graph)).values()) / num_nodes
        eigenvalues = sum(np.linalg.eigvals(nx.laplacian_matrix(graph).toarray())) / num_nodes
        density = nx.density(graph)
        diameter = nx.diameter(graph)
        average_cc = nx.average_clustering(graph)
        try:
            eigenvector_centrality = sum(dict(nx.eigenvector_centrality(graph)).values()) / num_nodes
        except:
            eigenvector_centrality = 0
        GB_aft_attr = torch.tensor([num_nodes, num_edges, average_cc, diameter, eigenvector_centrality, density, triangles, eigenvalues,edge_betweenness_centrality, closeness_centrality])
        return GB_aft_attr

def GB_coarsen(graph, node_tags, purity, degree_purity):
    labels = dict(zip(graph.nodes(), node_tags))
    nx.set_node_attributes(graph, labels, 'label')

    if nx.is_connected(graph):
        GB_graph, GB_node_tags = get_GB_graph(graph, purity, degree_purity)
    else:
        connected_components = list(nx.connected_components(graph))
        connected_subgraphs = [graph.subgraph(component) for component in connected_components]
        GB_graph_list = []
        GB_node_tags_list = []
        for connected_subgraph in connected_subgraphs:
            GB_graph, GB_node_tags = get_GB_graph(connected_subgraph, purity, degree_purity)

            GB_graph_list.append(GB_graph)
            GB_node_tags_list.extend(GB_node_tags)
        merged_graph = nx.Graph()
        for GB_graph in GB_graph_list:
            add_num = len(merged_graph) 
            GB_graph_add_num = nx.Graph()


            for node in GB_graph.nodes():
                new_node = node + add_num
                GB_graph_add_num.add_node(new_node, **GB_graph.nodes[node])  


            new_edges = [(edge[0] + add_num, edge[1] + add_num) for edge in GB_graph.edges()]
            GB_graph_add_num.add_edges_from(new_edges)


            merged_graph = nx.compose(merged_graph, GB_graph_add_num)

        GB_graph = merged_graph
        GB_node_tags = GB_node_tags_list
        
    return GB_graph, GB_node_tags