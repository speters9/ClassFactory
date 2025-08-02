"""Build and analyze concept maps from relationship data.

This module provides functionality to create, analyze and visualize concept maps
based on relationships between concepts extracted from educational content.

Functions:
    build_graph: Create a weighted graph from concept relationships.
    detect_communities: Identify concept clusters using various community detection algorithms.

The module supports both directed and undirected graphs, with features including:
- Edge weight normalization
- Node centrality calculation
- Community detection using multiple algorithms (leiden, louvain, spectral)
- Visualization preparation with node sizes and community labels
"""
# %%
import logging
# base libraries
import os
from pathlib import Path
from typing import List, Tuple

# graph setup
import networkx as nx
import networkx.algorithms.community as nx_comm
from cdlib import algorithms
# env setup
from dotenv import load_dotenv
from sklearn.cluster import SpectralClustering

# self-defined utils
from class_factory.concept_web.concept_extraction import process_relationships

load_dotenv()

# Path definitions
projectDir = Path(os.getenv('projectDir'))
dataDir = projectDir / "tests/data/"

# %%


def build_graph(
    processed_relationships: List[Tuple[str, str, str]],
    directed: bool = False
) -> nx.Graph | nx.DiGraph:
    """Build a graph from processed concept relationships.

    Args:
        processed_relationships (List[Tuple[str, str, str]]): List of (concept1, relationship, concept2) tuples
            representing relationships between concepts that have undergone entity resolution.
        directed (bool, optional): If True, creates a directed graph; if False, creates an undirected graph.
            Defaults to False.

    Returns:
        nx.Graph | nx.DiGraph: A NetworkX Graph or DiGraph with the following attributes:
            - Nodes have 'text_size' and 'centrality' attributes
            - Edges have 'weight', 'normalized_weight', and 'relation' attributes

    Raises:
        ValueError: If relationships are not correctly formatted.
    """
    # Initialize an undirected graph
    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes and edges from relationships
    for concept1, relationship, concept2 in processed_relationships:
        if relationship not in ["None", "none"]:
            if G.has_edge(concept1, concept2):
                # f"{concept1} -> {relationship} -> {concept2}")
                G[concept1][concept2]['relation'].add(relationship)
                G[concept1][concept2]['weight'] += 1
            else:
                # [f"{concept1} -> {relationship} -> {concept2}"])
                G.add_edge(concept1, concept2, weight=1,
                           relation={relationship})

    # Normalize edge weights and centrality
    edge_weights = nx.get_edge_attributes(G, 'weight').values()

    # Calculate min and max weights
    # Avoid division by zero
    max_weight = max(edge_weights) if edge_weights else 1
    # Avoid division by zero
    min_weight = min(edge_weights) if edge_weights else 1

    # Normalize edge weights
    min_normalized_weight = 0.5
    max_normalized_weight = 4

    try:
        for u, v, d in G.edges(data=True):
            normalized_weight = min_normalized_weight + (max_normalized_weight - min_normalized_weight) * \
                (d['weight'] - min_weight) / (max_weight - min_weight)
            G[u][v]['normalized_weight'] = normalized_weight

        # Calculate degree centrality for each node
        if directed:
            centrality = nx.in_degree_centrality(G)
        else:
            centrality = nx.degree_centrality(G)

        # Normalize centrality to a range suitable for text size (e.g., 10 to 50)
        min_size = 6
        max_size = 24
        max_centrality = max(centrality.values())
        min_centrality = min(centrality.values())

        for node, centrality_value in centrality.items():
            normalized_size = min_size + (max_size - min_size) * (
                centrality_value - min_centrality) / (max_centrality - min_centrality)
            G.nodes[node]['text_size'] = normalized_size
            G.nodes[node]['centrality'] = centrality_value

    except ZeroDivisionError:
        # Log a warning that the graph could not be normalized
        logging.warning(
            "Normalization of weights and centrality skipped due to lack of variation in the graph.\nReturning unnormalized edge weight and text size")
        # Fall back to default sizes if normalization fails
        for node in G.nodes():
            G.nodes[node]['text_size'] = 12  # Default text size
            G.nodes[node]['centrality'] = 0.5  # Default centrality

    return G


def detect_communities(
    G: nx.Graph | nx.DiGraph,
    method: str = "leiden",
    num_clusters: int | None = None
) -> nx.Graph | nx.DiGraph:
    """Detect communities in the concept graph.

    Args:
        G (nx.Graph | nx.DiGraph): The input graph for community detection.
        method (str, optional): Algorithm to use for community detection.
            Options: "leiden", "louvain", or "spectral". Defaults to "leiden".
        num_clusters (int | None, optional): Required number of clusters for spectral clustering.
            Only used when method="spectral". Defaults to None.

    Returns:
        nx.Graph | nx.DiGraph: A copy of the input graph with an additional 'community' node attribute
        indicating the community assignment for each node.

    Raises:
        ValueError: If the specified method is not "leiden", "louvain", or "spectral".
    """
    G_copy = G.copy()

    if method == "leiden":
        # Use Louvain method for community detection
        communities_obj = algorithms.leiden(G)
        # extract communities from 'nodeclustering' object
        communities = communities_obj.communities
    elif method == "louvain":
        # Use Louvain method for community detection
        communities = nx_comm.louvain_communities(G)
    elif method == "spectral":
        # Create a list of node names to maintain the order
        nodes = list(G.nodes())

        # Create the adjacency matrix for the graph
        adj_matrix = nx.to_numpy_array(G, nodelist=nodes)

        # Apply spectral clustering
        sc = SpectralClustering(n_clusters=num_clusters,
                                affinity='precomputed', assign_labels='kmeans')
        labels = sc.fit_predict(adj_matrix)

        # Group nodes by their cluster labels using node names instead of indices
        communities = [set() for _ in range(num_clusters)]
        for node, label in zip(nodes, labels):
            communities[label].add(node)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose 'louvain' or 'spectral'.")

    # Assign each node to its community for visualization
    for i, community in enumerate(communities):
        for node in community:
            # Assign a group ID for use in visualization
            G_copy.nodes[node]['community'] = i

    return G_copy


if __name__ == "__main__":
    import json

    with open(Path.home() / dataDir / 'conceptlist_test.json', 'r') as f:
        conceptlist = json.load(f)

    with open(Path.home() / dataDir / 'relationshiplist_test.json', 'r') as f:
        relationship_list = json.load(f)
    # Build the graph
    G_base = build_graph(relationship_list, directed=True)
    # Detect communities using Louvain method
    G = detect_communities(G_base, method="leiden")

# %%
