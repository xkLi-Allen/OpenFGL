import networkx as nx
import torch
import torch_geometric
import torch_geometric.data
from torch_geometric.utils import to_networkx, to_dgl
from torch_geometric.utils import degree as pyg_degree
from torch_geometric.utils import homophily as pyg_homophily
import numpy as np
import dgl 
# pytroch2.1.*+dgl_cu11.8

# graph-level 

def average_kl_divergence(label_distributions):
    """
    Computes the average Kullback-Leibler divergence between individual label distributions and the global label distribution.

    Args:
        label_distributions (np.array): An array where each row represents a label distribution for a node.

    Returns:
        float: The average KL divergence across all nodes.
    """
    global_dist = np.mean(label_distributions, axis=0)
    kl_div = np.sum(label_distributions * np.log((label_distributions + 1e-9) / (global_dist + 1e-9)), axis=1)
    return np.mean(kl_div)

def degree_distribution(data: torch_geometric.data.Data) -> list:
    """
    Computes the degree distribution of the nodes in a graph.

    Args:
        data (torch_geometric.data.Data): A graph represented as a PyTorch Geometric data object.

    Returns:
        list: List of degrees of all nodes in the graph.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    degrees = [degree for node, degree in graph_nx.degree()]
    return degrees

def degree_kurtosis(data: torch_geometric.data.Data):
    """
    Calculates the kurtosis of the degree distribution of the graph. Kurtosis indicates the shape of the distribution.

    Args:
        data (torch_geometric.data.Data): A graph represented as a PyTorch Geometric data object.

    Returns:
        float: Kurtosis of the degree distribution.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    degrees = [degree for node, degree in graph_nx.degree()]
    kurtosis = np.mean((degrees - np.mean(degrees))**4) / (np.std(degrees)**4) - 3
    return kurtosis

def degree_mean(data: torch_geometric.data.Data):
    """
    Calculates the mean of the degree distribution for the graph.

    Args:
        data (torch_geometric.data.Data): A graph represented as a PyTorch Geometric data object.

    Returns:
        float: Mean degree of the nodes in the graph.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    degrees = [degree for node, degree in graph_nx.degree()]
    degrees_mean = np.mean(degrees)
    return degrees_mean

def degree_variance(data: torch_geometric.data.Data):
    """
    Calculates the variance of the degree distribution of the graph.

    Args:
        data (torch_geometric.data.Data): A graph represented as a PyTorch Geometric data object.

    Returns:
        float: Variance of the degree distribution.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    degrees = [degree for node, degree in graph_nx.degree()]
    degrees_var = np.var(degrees)
    return degrees_var

def degree_centrality(data: torch_geometric.data.Data) -> dict:
    """
    Computes the degree centrality for each node in the graph.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        dict: A dictionary mapping each node to its degree centrality.
    """
    
    graph_nx = to_networkx(data, to_undirected=True)
    degree_centrality = nx.degree_centrality(graph_nx)
    return degree_centrality

def closeness_centrality(data: torch_geometric.data.Data, u=None):
    """
    Computes the closeness centrality for all nodes or a specific node in the graph.

    Args:
        data (torch_geometric.data.Data): The graph data.
        u (int, optional): The node for which to calculate closeness centrality. If None, calculates for all nodes.

    Returns:
        dict or float: Closeness centrality of all nodes or of the specified node.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    if u is None:
        closeness_centrality = nx.closeness_centrality(graph_nx)
        return closeness_centrality
    else:
        closeness_centrality = nx.closeness_centrality(graph_nx, u)
        return closeness_centrality
    
def load_centrality(data: torch_geometric.data.Data):
    """
    Placeholder for loading a centrality measure which is not implemented.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Raises:
        NotImplementedError: Indicates that this function is a placeholder and needs an implementation.
    """
    raise NotImplementedError

def eigenvector_centrality(data: torch_geometric.data.Data):
    """
    Placeholder for computing eigenvector centrality, which is not implemented.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Raises:
        NotImplementedError: Indicates that this function is a placeholder and needs an implementation.
    """
    raise NotImplementedError

def degree_assortativity_coefficient(data: torch_geometric.data.Data):
    """
    Computes the Pearson correlation coefficient of node degrees in the graph.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        float: Degree Pearson correlation coefficient.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    degree_assortativity_coefficient = nx.degree_assortativity_coefficient(graph_nx)
    return degree_assortativity_coefficient

def degree_pearson_correlation_coefficient(data: torch_geometric.data.Data):
    graph_nx = to_networkx(data, to_undirected=True)
    degree_pearson_correlation_coefficient = nx.degree_pearson_correlation_coefficient(graph_nx)
    return degree_pearson_correlation_coefficient

def average_degree_connectivity(data: torch_geometric.data.Data, weight=None):
    """
    Computes the average degree connectivity of the graph.

    Args:
        data (torch_geometric.data.Data): The graph data.
        weight (str, optional): The weight attribute of the edges to consider in the calculation.

    Returns:
        dict: Average degree connectivity for each degree value.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    if weight is None:
        average_degree_connectivity = nx.average_degree_connectivity(graph_nx)
        return average_degree_connectivity
    else:
        average_degree_connectivity = nx.average_degree_connectivity(graph_nx, weight=weight)
        return average_degree_connectivity

def clustering_coefficient(data: torch_geometric.data.Data):
    """
    Computes the clustering coefficient for each node in the graph.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        dict: Clustering coefficients for each node.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    clustering_coeffs = nx.clustering(graph_nx)
    return clustering_coeffs

def avg_clustering_coefficient(data: torch_geometric.data.Data):
    """
    Computes the average clustering coefficient of the graph.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        float: Average clustering coefficient.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    avg_clustering_coeffs = nx.average_clustering(graph_nx)
    return avg_clustering_coeffs    

def avg_shortest_path_length(data: torch_geometric.data.Data):
    """
    Computes the average shortest path length in the graph or its largest connected component if the graph is not connected.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        float: Average shortest path length.
    """
    graph_nx = to_networkx(data, to_undirected=True)

    if nx.is_connected(graph_nx):
        avg_shortest_path_length = nx.average_shortest_path_length(graph_nx)
    else:
        largest_cc = max(nx.connected_components(graph_nx), key=len)
        subgraph = graph_nx.subgraph(largest_cc)
        avg_shortest_path_length = nx.average_shortest_path_length(subgraph)
    
    return avg_shortest_path_length

def largest_component_percentage(data: torch_geometric.data.Data):
    """
    Calculates the percentage of nodes in the largest connected component of the graph relative to the total number of nodes.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        float: Percentage of nodes in the largest connected component.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    
    largest_cc_size = len(max(nx.connected_components(graph_nx), key=len))
    largest_component_percentage = (largest_cc_size / len(graph_nx.nodes())) * 100

    return largest_component_percentage

def avg_local_efficiency(data: torch_geometric.data.Data):
    """
    Computes the average local efficiency of the graph.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        float: Average local efficiency.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    avg_local_efficiency = nx.local_efficiency(graph_nx)
    return avg_local_efficiency

def avg_global_efficiency(data: torch_geometric.data.Data):
    """
    Computes the average global efficiency of the graph.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        float: Average global efficiency.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    avg_global_efficiency = nx.global_efficiency(graph_nx)
    return avg_global_efficiency

def diameter(data: torch_geometric.data.Data):
    """
    Computes the diameter of the graph or of each connected component if the graph is not connected.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        int or list: Diameter of the graph or a list of diameters for each connected component.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    if nx.is_connected(graph_nx):
        diameter = nx.diameter(graph_nx)
        return diameter
    else:
        diameter_list = []
        connected_components = nx.connected_components(graph_nx)
        for i, component in enumerate(connected_components):
            subgraph = graph_nx.subgraph(component)
            diameter = nx.diameter(subgraph)
            diameter_list.append(diameter)
        return diameter_list

def transitivity(data: torch_geometric.data.Data):
    """
    Computes the transitivity of the graph, which is the overall probability that two adjacent vertices of a vertex are connected.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        float: Transitivity of the graph.
    """
    graph_nx = to_networkx(data, to_undirected=True)
    transitivity = nx.transitivity(graph_nx)
    return transitivity

# subgraph
def label_distribution(data: torch_geometric.data.Data):
    """
    Counts the distribution of labels across nodes in a graph.

    Args:
        data (torch_geometric.data.Data): A graph represented as a PyTorch Geometric data object.

    Returns:
        collections.Counter: A Counter object counting the occurrence of each label.
    """
    from collections import Counter
    labels = data.y.tolist()
    label_distribution = Counter(labels)
    return label_distribution

def homophily(data: torch_geometric.data.Data, method=None):
    """
    Computes the homophily ratio of a graph based on given method, which describes the tendency of nodes to connect to other nodes with the same label.

    Args:
        data (torch_geometric.data.Data): A graph represented as a PyTorch Geometric data object.
        method (str): Specifies the method for calculating homophily ('node', 'edge', 'edge_insensitive', 'adjusted').

    Returns:
        float: Homophily ratio according to the specified method.
    """
    # ICLR'20 Geom-gcn: Geometric graph convolutional networks
    # NeurIPS'20 Beyond homophily in graph neural networks: Current limitations and effective designs
    # NeurIPS'21 Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods
    # ICLR'23 A critical look at the evaluation of gnns under heterophily: are we really making progress?
    assert method in ['node', 'edge', 'edge_insensitive', 'adjusted']
    if method in ['node', 'edge', 'edge_insensitive']:
        edge_index = data.edge_index
        labels = data.y
        homo_score = pyg_homophily(edge_index, labels, method=method)
        return homo_score
    elif method in ['adjusted',]:
        from collections import Counter
        edge_index = data.edge_index
        labels = data.y
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        homo_score = pyg_homophily(edge_index, labels, method='edge')
        label_distribution = Counter(labels.tolist())
        degrees = pyg_degree(edge_index[0])
        degree_dict = {label: 0 for label in label_distribution.keys()}
        for nd in range(num_nodes):
            degree_dict[labels[nd].item()] += degrees[nd].item()

        sum_p2 = 0.
        for label in label_distribution.keys():
            # 对于无向图，边数的定义
            sum_p2 += degree_dict[label] ** 2 / (2*(num_edges / 2)) ** 2
        adjusted_homo_score = (homo_score - sum_p2) / (1 - sum_p2)
        return adjusted_homo_score

def aggregation_similarity(data: torch_geometric.data.Data, modified=None):
    """
    Placeholder for computing aggregation similarity for graph neural networks. Requires implementation specific to model and dataset.

    Args:
        data (torch_geometric.data.Data): A graph represented as a PyTorch Geometric data object.
        modified (bool): Additional flags or parameters that could affect the computation.

    Raises:
        NotImplementedError: Indicates that the function needs a user-provided implementation.
    """
    # NeurIPS'23 Revisiting Heterophily For Graph Neural Networks
    # need a model
    raise NotImplementedError

def label_informativeness(data: torch_geometric.data.Data, method=None):
    """
    Computes the informativeness of labels on a graph based on node or edge context, which helps in understanding how well labels relate to the graph structure.

    Args:
        data (torch_geometric.data.Data): The graph data.
        method (str, optional): The method used for computing informativeness, either 'node' or 'edge'.

    Returns:
        float: The informativeness score, which quantifies the relevance of labels to their graph context.
    """
    # NeurIPS'23 Characterizing graph datasets for node classification: Beyond homophily-heterophily dichotomy
    assert method in ['node', 'edge']
    labels = data.y
    dgl_g = to_dgl(data)
    if method == 'node':
        label_informativeness = dgl.node_label_informativeness(dgl_g, labels)
    elif method == 'edge':
        label_informativeness = dgl.edge_label_informativeness(dgl_g, labels)
    return label_informativeness

def feature_sparsity(data: torch_geometric.data.Data):
    """
    Computes the sparsity of features in the graph by calculating the ratio of non-zero features to the total number of features.

    Args:
        data (torch_geometric.data.Data): The graph data containing node features.

    Returns:
        float: The ratio of non-zero features to total features, indicating the sparsity of the feature set.
    """
    features = data.x
    num_nonzero = torch.count_nonzero(features).item()
    return num_nonzero / torch.numel(features)

def edge_sparsity(data: torch_geometric.data.Data):
    """
    Computes the sparsity of edges in the graph, providing an understanding of the graph's connectivity.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        float: The ratio of the actual number of edges to the maximum possible number of edges, indicating how sparsely the nodes are connected.
    """
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    # undirected graph
    return num_edges / (num_nodes * (num_nodes - 1))