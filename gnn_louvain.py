import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
import community as community_louvain  # For Louvain
import leidenalg as la  # Leiden algorithm
import igraph as ig
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from cdlib import evaluation
from torch_geometric.nn import TransformerConv
from cdlib import NodeClustering
from numpy.linalg import norm
import torch.nn as nn
from GNN import GraphTransformer, GCN
import argparse
import time
# Define a simple Graph Transformer model




# Step 1: Generate Node Embeddings using Graph Transformer
def get_graph_transformer_embeddings(graph, features, hidden_channels=64, epochs=200, num_heads=4):
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)

    model = GraphTransformer(x.size(1), hidden_channels, num_heads=num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.mse_loss(out, x)  # You can change the loss based on your task
        loss.backward()
        optimizer.step()

    return out.detach().numpy()

# Define a simple GCN model


# Step 1: Generate Node Embeddings using GCN
def get_node_embeddings(graph, features, hidden_channels=64, epochs=200, dropout=0.5):
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)

    model = GCN(x.size(1), hidden_channels, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.mse_loss(out, x)  # You can change the loss based on your task
        loss.backward()
        optimizer.step()

    return out.detach().numpy()


# Step 2: Apply Louvain Algorithm
def apply_louvain(graph):
    partition_dict = community_louvain.best_partition(graph, weight='weight')
    communities = {}
    for node, community in partition_dict.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    partition = [sorted(members) for members in communities.values()]
    partition.sort(key=lambda x: x[0])
    return partition

# Step 3: Apply Leiden Algorithm
def apply_leiden(graph):
    edges = list(graph.edges())
    g = ig.Graph(edges=edges)
    weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    partition = la.find_partition(g, la.ModularityVertexPartition, weights=weights)
    return [sorted(community) for community in partition]

# Step 4: Calculate Modularity
def calculate_modularity(graph, partition):
    g = ig.Graph(edges=list(graph.edges()))
    community_map = [0] * graph.number_of_nodes()
    for i, community in enumerate(partition):
        for node in community:
            community_map[node] = i
    return g.modularity(community_map)

# Step 5: Calculate Evaluation Metrics (NMI, ARI, F-Score)
def calculate_metrics(partition, ground_truth):
    pred_labels = np.zeros(len(ground_truth))
    for community_idx, community in enumerate(partition):
        for node in community:
            pred_labels[node] = community_idx

    nmi = NMI(ground_truth, pred_labels)
    ari = ARI(ground_truth, pred_labels)

    f_score = f1_score(ground_truth, pred_labels, average='macro')

    return nmi, ari, f_score


from sklearn.metrics.pairwise import cosine_similarity

# Step 2: Compute Similarity for each edge and assign weights
def compute_similarity(graph, embeddings):
    similarities = {}
    # Compute the pairwise cosine similarities for all embeddings
    cosine_sim_matrix = cosine_similarity(embeddings)

    for u, v in graph.edges():
        sim = cosine_sim_matrix[u][v]  # Get the cosine similarity between nodes u and v
        similarities[(u, v)] = sim

    return similarities


def get_top_k_neighbors(similarity_dict, num_nodes, k=5):
    node_neighbors = {i: [] for i in range(num_nodes)}
    similarity_matrix = np.zeros((num_nodes, num_nodes))

    # Create a similarity matrix using the computed similarities
    for (u, v), sim in similarity_dict.items():
        similarity_matrix[u][v] = sim
        similarity_matrix[v][u] = sim  # Since the graph is undirected

    top_k_neighbors = {}
    farthest_neighbors = {}

    # For each node, sort neighbors by similarity and pick top-k and farthest-k
    for i in range(num_nodes):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]  # Sort in descending order
        top_k_neighbors[i] = sorted_indices[1:k+1]  # Top k nearest neighbors (excluding itself)
        farthest_neighbors[i] = sorted_indices[-k:]  # Farthest neighbors

    return top_k_neighbors, farthest_neighbors

# Step 4: Rewire Edges Based on Top-k Neighbors
def rewire_edges(graph, top_k_neighbors, farthest_neighbors):
    new_graph = graph.copy()

    # Add edges for top-k nearest neighbors
    # for node, neighbors in top_k_neighbors.items():
    #     for neighbor in neighbors:
    #         if not new_graph.has_edge(node, neighbor):
    #             new_graph.add_edge(node, neighbor)

    # Remove edges for k farthest neighbors
    for node, farthest in farthest_neighbors.items():
        for neighbor in farthest:
            if new_graph.has_edge(node, neighbor):
                new_graph.remove_edge(node, neighbor)

    return new_graph

# Integrate edge rewiring after embedding generation
def apply_edge_rewiring(graph, embeddings, k=5):
    similarity_dict = compute_similarity(graph, embeddings)
    top_k_neighbors, farthest_neighbors = get_top_k_neighbors(similarity_dict, graph.number_of_nodes(), k=k)
    new_graph = rewire_edges(graph, top_k_neighbors, farthest_neighbors)
    return new_graph


def average_neighbour_distance(u, v, graph, distance_matrix):
    """
    Compute the average distance between the neighbours of node u and node v.

    Parameters:
    u, v: nodes in the graph
    graph: the graph itself (networkx)
    distance_matrix: a matrix where distance_matrix[i][j] represents the distance between node i and node j

    Returns:
    avg_distance: the average distance between the neighbours of u and v
    """
    # Get neighbours of u and v
    neighbours_u = list(graph.neighbors(u))
    neighbours_v = list(graph.neighbors(v))

    # Calculate the average pairwise distance between neighbours of u and neighbours of v
    total_distance = 0
    count = 0
    for neighbour_u in neighbours_u:
        for neighbour_v in neighbours_v:
            total_distance += distance_matrix[neighbour_u][neighbour_v]
            count += 1

    if count == 0:
        return float('inf')  # If no neighbours, return a large value

    avg_distance = total_distance / count

    return avg_distance


def rewire_graph2(graph, embeddings, k=3):
    # Step 1: Compute pairwise cosine distance matrix from the embeddings
    cosine_sim_matrix = cosine_similarity(embeddings)
    cosine_dist_matrix = 1 - cosine_sim_matrix  # Cosine distance = 1 - Cosine similarity

    num_nodes = graph.number_of_nodes()
    non_existing_edges = []
    existing_edges = []

    # Step 2: Compute average neighbour distances between all node pairs
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):  # Avoid duplicate pairs (u, v) and (v, u)
            avg_neighbour_distance = average_neighbour_distance(u, v, graph, cosine_dist_matrix)

            if not graph.has_edge(u, v):
                # Store non-existing edge with its average distance
                non_existing_edges.append((u, v, avg_neighbour_distance))
            else:
                # Store existing edge with its average distance
                existing_edges.append((u, v, avg_neighbour_distance))

    # Step 3: Sort non-existing edges by average distance (ascending) - closest first
    non_existing_edges.sort(key=lambda x: x[2])

    # Step 4: Get top k non-existing closest edges and add them
    top_k_non_existing = non_existing_edges[:k]
    for (u, v, distance) in top_k_non_existing:
        if not graph.has_edge(u, v):
            graph.add_edge(u, v, weight=cosine_sim_matrix[u][v])  # Add edge with cosine similarity as weight
            print(f"Added edge between {u} and {v}, distance: {distance}, weight: {cosine_sim_matrix[u][v]}")

    # Step 5: Sort existing edges by average distance (descending) - farthest first
    existing_edges.sort(key=lambda x: x[2], reverse=True)

    # Step 6: Get top k existing farthest edges and remove them
    #top_k_existing_to_remove = existing_edges[:k]
    #for (u, v, distance) in top_k_existing_to_remove:
     #   if graph.has_edge(u, v):
    #        graph.remove_edge(u, v)
     #       print(f"Removed edge between {u} and {v}, distance: {distance}")

    #return top_k_non_existing, top_k_existing_to_remove  # Return the added and removed edges if needed


def rewire_graph(graph, embeddings, k=3):
    # Compute cosine similarity for all pairs of embeddings
    cosine_sim_matrix = cosine_similarity(embeddings)


    # Convert cosine similarity to cosine distance
    cosine_dist_matrix = 1 - cosine_sim_matrix  # Cosine distance = 1 - Cosine similarity

    num_nodes = graph.number_of_nodes()

    for i in range(num_nodes):
        # Get top-k closest and farthest nodes for node i
        closest_nodes = np.argsort(cosine_dist_matrix[i])[1:k + 1]  # Ignore self, get top-k closest nodes based on cosine distance
        farthest_nodes = np.argsort(cosine_dist_matrix[i])[-k:]  # Get the k farthest nodes

        # Add edges for closest nodes if they are not connected, and set the weight as similarity (1 - distance)
        for node in closest_nodes:
            if not graph.has_edge(i, node):
               # print('an edge is added')
                graph.add_edge(i, node, weight=cosine_sim_matrix[i][node])  # Use cosine similarity as the edge weight
                print(graph.nodes[i]['club'])
                print(graph.nodes[node]['club'])
#        ground_truth[i] = G.nodes[i]['club'] == 'Mr. Hi'  # Example from Karate Club dataset

        # Remove edges for farthest nodes if they are connected
        #for node in farthest_nodes:
        #    if graph.has_edge(i, node):
        #       print('an edge is removed')
               # graph.remove_edge(i, node)

def convert_to_node_clustering(partition, graph):
    # Convert the partition (list of lists) to a CDlib NodeClustering object
    return NodeClustering(
        communities=partition,
        graph=graph,
        method_name="custom"
    )

def amplify_power(x, power):
    return x ** power

def sigmoid_amplify(x, k=1):
    return 1 / (1 + np.exp(-k * x))

def assign_weights(graph, similarities):

    for (u, v), sim in similarities.items():
        graph[u][v]['weight'] = sigmoid_amplify(sim,1)


def calculate_cdlib_metrics(graph, partition, ground_truth_partition):
    # Convert partitions to CDlib's NodeClustering format
    clustering = convert_to_node_clustering(partition, graph)
    ground_truth_clustering = convert_to_node_clustering(ground_truth_partition, graph)

    # Calculate CDlib metrics
    nf1 = evaluation.nf1(clustering, ground_truth_clustering).score
    southwood = evaluation.southwood_index(clustering, ground_truth_clustering)
    rogers_tanimoto = evaluation.rogers_tanimoto_index(clustering, ground_truth_clustering)
    sorensen = evaluation.sorensen_index(clustering, ground_truth_clustering)
    dice = evaluation.dice_index(clustering, ground_truth_clustering)

    return nf1, southwood, rogers_tanimoto, sorensen, dice
def compare_methods(graph, features, ground_truth, num_trials=5, k=5 ):
    # Initialize lists to store the results of each trial
    results = []

    # Ground truth as a partition (list of lists)
    ground_truth_partition = [list(np.where(np.array(ground_truth) == i)[0]) for i in np.unique(ground_truth)]

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}...")

        # Apply Louvain without GCN
        start_time = time.time()
        louvain_partition = apply_louvain(graph)
        end_time = time.time()
        louvain_modularity = calculate_modularity(graph, louvain_partition)
        louvain_nmi, louvain_ari, louvain_fscore = calculate_metrics(louvain_partition, ground_truth)
        louvain_num_communities = len(louvain_partition)
        print("Louvain time:", end_time - start_time)

        # Generate node embeddings using GCN
        start_time = time.time()
        embeddings_gcn = get_node_embeddings(graph, features)
        end_time = time.time()
        embeddings_time = end_time - start_time

        # Apply GCN + Louvain
        start_time = time.time()
        nG = graph.copy()
        assign_weights(nG, compute_similarity(graph, embeddings_gcn))
        gcn_louvain_partition = apply_louvain(nG)
        end_time = time.time()
        gcn_louvain_modularity = calculate_modularity(graph, gcn_louvain_partition)
        gcn_louvain_nmi, gcn_louvain_ari, gcn_louvain_fscore = calculate_metrics(gcn_louvain_partition, ground_truth)
        gcn_louvain_num_communities = len(gcn_louvain_partition)
        print("GCN time:", end_time - start_time+embeddings_time)

        # Rewire graph based on GCN embeddings and apply Louvain
        start_time = time.time()
        rewired_graph_gcn = graph.copy()
        rewire_graph2(rewired_graph_gcn, embeddings_gcn, k)
        assign_weights(rewired_graph_gcn, compute_similarity(rewired_graph_gcn, embeddings_gcn))
        gcn_louvain_rewire_partition = apply_louvain(rewired_graph_gcn)
        end_time = time.time()
        gcn_louvain_rewire_modularity = calculate_modularity(graph, gcn_louvain_rewire_partition)
        gcn_louvain_rewire_nmi, gcn_louvain_rewire_ari, gcn_louvain_rewire_fscore = calculate_metrics(
            gcn_louvain_rewire_partition, ground_truth)
        gcn_louvain_rewire_num_communities = len(gcn_louvain_rewire_partition)
        print("GCN rewire time:", end_time - start_time + embeddings_time)


        # Generate node embeddings using Graph Transformer
        start_time = time.time()
        embeddings_transformer = get_graph_transformer_embeddings(graph, features)
        end_time = time.time()
        embeddings_time = end_time - start_time

        # Apply Graph Transformer + Louvain
        start_time = time.time()
        nG = graph.copy()
        assign_weights(nG, compute_similarity(graph, embeddings_transformer))
        transformer_louvain_partition = apply_louvain(nG)
        end_time = time.time()
        transformer_louvain_modularity = calculate_modularity(graph, transformer_louvain_partition)
        transformer_louvain_nmi, transformer_louvain_ari, transformer_louvain_fscore = calculate_metrics(
            transformer_louvain_partition, ground_truth)
        transformer_louvain_num_communities = len(transformer_louvain_partition)
        print("Transformer time:", end_time - start_time+embeddings_time)


        # Rewire graph based on Transformer embeddings and apply Louvain
        start_time = time.time()
        rewired_graph_transformer = graph.copy()
        rewire_graph2(rewired_graph_transformer, embeddings_transformer, k)
        assign_weights(rewired_graph_transformer, compute_similarity(rewired_graph_transformer, embeddings_transformer))
        transformer_louvain_rewire_partition = apply_louvain(rewired_graph_transformer)
        end_time = time.time()
        transformer_louvain_rewire_modularity = calculate_modularity(graph, transformer_louvain_rewire_partition)
        transformer_louvain_rewire_nmi, transformer_louvain_rewire_ari, transformer_louvain_rewire_fscore = calculate_metrics(
            transformer_louvain_rewire_partition, ground_truth)
        transformer_louvain_rewire_num_communities = len(transformer_louvain_rewire_partition)
        print("Transformer rewire time:", end_time - start_time+embeddings_time)


        # Collect results for this trial
        results.append({
            'Trial': trial + 1,
            'Louvain_Modularity': louvain_modularity,
            'GCN_Louvain_Modularity': gcn_louvain_modularity,
            'Transformer_Louvain_Modularity': transformer_louvain_modularity,
            'GCN_Louvain_Rewire_Modularity': gcn_louvain_rewire_modularity,
            'Transformer_Louvain_Rewire_Modularity': transformer_louvain_rewire_modularity,
            'Louvain_NMI': louvain_nmi,
            'GCN_Louvain_NMI': gcn_louvain_nmi,
            'Transformer_Louvain_NMI': transformer_louvain_nmi,
            'GCN_Louvain_Rewire_NMI': gcn_louvain_rewire_nmi,
            'Transformer_Louvain_Rewire_NMI': transformer_louvain_rewire_nmi,
            'Louvain_ARI': louvain_ari,
            'GCN_Louvain_ARI': gcn_louvain_ari,
            'Transformer_Louvain_ARI': transformer_louvain_ari,
            'GCN_Louvain_Rewire_ARI': gcn_louvain_rewire_ari,
            'Transformer_Louvain_Rewire_ARI': transformer_louvain_rewire_ari,
            'Louvain_F-Score': louvain_fscore,
            'GCN_Louvain_F-Score': gcn_louvain_fscore,
            'Transformer_Louvain_F-Score': transformer_louvain_fscore,
            'GCN_Louvain_Rewire_F-Score': gcn_louvain_rewire_fscore,
            'Transformer_Louvain_Rewire_F-Score': transformer_louvain_rewire_fscore,
            'Louvain_Num_Communities': louvain_num_communities,
            'GCN_Louvain_Num_Communities': gcn_louvain_num_communities,
            'Transformer_Louvain_Num_Communities': transformer_louvain_num_communities,
            'GCN_Louvain_Rewire_Num_Communities': gcn_louvain_rewire_num_communities,
            'Transformer_Louvain_Rewire_Num_Communities': transformer_louvain_rewire_num_communities
        })

    # Convert results to a DataFrame for easier viewing
    results_df = pd.DataFrame(results)

    # Calculate mean and standard deviation
    summary_df = pd.DataFrame({
        'Metric': results_df.columns[1:],  # Skip 'Trial' column
        'Mean': results_df.iloc[:, 1:].mean(),
        'Standard Deviation': results_df.iloc[:, 1:].std()
    })

    # Print results for each trial as a table
    print("\nResults for each trial:")
    print(results_df.to_string(index=False))

    # Print average and standard deviation
    print("\nSummary of Averages and Standard Deviations:")
    print(summary_df.to_string(index=False))

    return results_df, summary_df


def GCN_louvain(graph, features, ground_truth, num_trials=5):
    # Initialize lists to store the results of each trial
    results = []

    # Ground truth as a partition (list of lists)
    ground_truth_partition = [list(np.where(np.array(ground_truth) == i)[0]) for i in np.unique(ground_truth)]

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}...")

        # Generate node embeddings using GCN
        embeddings_gcn = get_node_embeddings(graph, features,dropout=dropout)

        # Apply GCN + Louvain
        assign_weights(graph, compute_similarity(graph, embeddings_gcn))
        gcn_louvain_partition = apply_louvain(graph)
        gcn_louvain_modularity = calculate_modularity(graph, gcn_louvain_partition)
        gcn_louvain_nmi, gcn_louvain_ari, gcn_louvain_fscore = calculate_metrics(gcn_louvain_partition, ground_truth)
        gcn_louvain_num_communities = len(gcn_louvain_partition)
        gcn_louvain_nf1, gcn_louvain_southwood, gcn_louvain_rogers_tanimoto, gcn_louvain_sorensen, gcn_louvain_dice = [
            metric[0] if isinstance(metric, tuple) else metric for metric in calculate_cdlib_metrics(graph, gcn_louvain_partition, ground_truth_partition)
        ]


        # Collect results for this trial
        results.append({
            'Trial': trial + 1,
            'GCN_Louvain_Modularity': gcn_louvain_modularity,
            'GCN_Louvain_NMI': gcn_louvain_nmi,
            'GCN_Louvain_ARI': gcn_louvain_ari,
            'GCN_Louvain_F-Score': gcn_louvain_fscore,
            'GCN_Louvain_NF1': gcn_louvain_nf1,
            'GCN_Louvain_Southwood': gcn_louvain_southwood,
            'GCN_Louvain_Rogers-Tanimoto': gcn_louvain_rogers_tanimoto,
            'GCN_Louvain_Sorensen': gcn_louvain_sorensen,
            'GCN_Louvain_Dice': gcn_louvain_dice,
            'GCN_Louvain_Num_Communities': gcn_louvain_num_communities
        })

    # Convert results to a DataFrame for easier viewing
    results_df = pd.DataFrame(results)

    # Calculate mean and standard deviation
    summary_df = pd.DataFrame({
        'Metric': results_df.columns[1:],  # Skip 'Trial' column
        'Mean': results_df.iloc[:, 1:].mean(),
        'Standard Deviation': results_df.iloc[:, 1:].std()
    })

    # Print results for each trial as a table
    print("\nResults for each trial:")
    print(results_df.to_string(index=False))

    # Print average and standard deviation
    print("\nSummary of Averages and Standard Deviations:")
    print(summary_df.to_string(index=False))

    return results_df, summary_df



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='GNN Louvain')
    parser.add_argument('--network', type=str, default='./dataset/karate', help='Path to the edgelist file')
    parser.add_argument('--method', type=str, default='compare', help='Method to use (gcn or transformer)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of trials to run')
    parser.add_argument('--node_features', type=int, default=64, help='Number of node features')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()
    filepath = args.network + '/network.dat'
    method = args.method
    dropout = args.dropout
    num_trials = args.num_trials
    num_node_features = args.node_features
    learning_rate = args.learning_rate

    GT_file_path = args.network + '/community.dat'
    # Load your graph
    G = nx.read_edgelist(filepath, nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))

    print(filepath, "dropout:", dropout, "num_trials:", num_trials, "num_node_features:", num_node_features, "learning_rate:", learning_rate)

    # Generate random node features (Replace with actual features)

    features = np.random.rand(G.number_of_nodes(), num_node_features)

    # Ground truth communities (Replace with actual labels if available)
    ground_truth = [0] * G.number_of_nodes()
    with open(GT_file_path) as f:
        for line in f:
            node, community = line.strip().split()
            ground_truth[int(node)] = int(community)

    # Compare methods
    if method == "compare":
        compare_methods(G, features, ground_truth, num_trials=num_trials, k=5)
    elif method == "gcn":
        GCN_louvain(G, features, ground_truth, num_trials=num_trials)
