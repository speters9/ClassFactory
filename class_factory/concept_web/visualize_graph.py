
"""
This module provides functions to visualize a concept map generated from processed relationships between concepts.
It includes functionalities to create interactive graph visualizations and generate word clouds representing the concepts.

The primary functionalities include:
1. **Interactive Graph Visualization**: Converts a NetworkX graph into an interactive HTML visualization using pyvis.
   The graph can be manipulated dynamically in a web browser, allowing for physics simulations, node filtering, and clustering.
2. **Word Cloud Generation**: Creates a word cloud image from a list of concepts, visually representing the frequency
   of each concept.

Main Functions:
- `visualize_graph_interactive(G: nx.Graph, output_path: Union[Path, str]) -> None`:
    Visualizes the given graph interactively using pyvis and saves the result as an HTML file. The nodes are colored
    based on their community, and the visualization allows for interactive exploration of the graph.

Workflow:
1. **Graph Conversion**: Converts the provided NetworkX graph into a pyvis graph, applying styles and attributes
   like node size and edge width based on centrality and relationship frequency.
2. **Interactive Visualization**: Saves the interactive graph as an HTML file, which can be explored in any web browser.

Dependencies:
- NetworkX: For graph data structure and manipulation.
- Matplotlib: For color mapping and displaying the word cloud.
- Pyvis: For creating interactive graph visualizations in HTML.
"""

from pathlib import Path
from typing import List, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network


def visualize_graph_interactive(
    G: nx.Graph,
    output_path: Union[Path, str],
    directed: bool = False,
    dark_mode: bool = True,
    max_nodes: int = 250,
    centrality_method: str = "degree",
    expand_neighbors: bool = True,
) -> None:
    """
    Create an interactive HTML visualization of a concept map using pyvis.

    Args:
        G (nx.Graph): The graph to visualize (with community and text_size attributes).
        output_path (Union[Path, str]): Path to save the HTML file.
        directed (bool, optional): If True, show edge arrows. Defaults to False.
        dark_mode (bool, optional): Use dark background. Defaults to True.
    """
    # Filter large graphs
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        G = filter_graph_by_centrality(
            G,
            max_nodes=max_nodes,
            method=centrality_method,
            expand_neighbors=expand_neighbors,
        )

    if dark_mode:
        net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', directed=directed)
    else:
        net = Network(height='750px', width='100%', bgcolor='white', font_color='black', directed=directed)

    # Generate a color map based on the number of communities
    communities = set(nx.get_node_attributes(G, 'community').values())
    color_map = plt.colormaps['tab20']  # 'tab20' is a colormap with 20 distinct colors
    community_colors = {community: mcolors.to_hex(color_map(i)) for i, community in enumerate(communities)}

    # Assign colors to nodes based on their community group
    for node in G.nodes():
        community = G.nodes[node]['community']
        G.nodes[node]['color'] = community_colors[community]  # Set the color attribute

    # Convert the NetworkX graph to a pyvis graph and add text size
    net.from_nx(G)
    for node in net.nodes:
        node["size"] = node.get('text_size', 20)
        node["font"].update({"size": node.get('text_size', 20)})

    for edge in net.edges:
        edge['relation'] = list(edge['relation'])
        edge['title'] = ", ".join(edge['relation'])
        edge['width'] = edge.get('normalized_weight', edge.get('weight', 1))
        if directed:
            edge["arrows"] = "to"

    # Add physics controls for a dynamic layout
    net.show_buttons(filter_=['layout',
                              'physics'])

    output_path = str(output_path)
    net.save_graph(output_path)
    print(f"Concept map saved to {output_path}")

    # Optionally, you can also open it directly in a browser
    # net.show(output_path)


def filter_graph_by_centrality(
    G: nx.Graph,
    max_nodes: int = 250,
    method: str = "pagerank",
    expand_neighbors: bool = True,
) -> nx.Graph:
    """
    Return an induced subgraph containing up to `max_nodes` most-central nodes.

    Parameters:
        G: original NetworkX graph
        max_nodes: desired maximum number of nodes in the returned graph
        method: centrality method to rank nodes ('pagerank', 'degree', 'betweenness')
        expand_neighbors: if True, after selecting top central nodes, try to include
                          their 1-hop neighbors to preserve local context until max_nodes reached.

    Returns:
        A copy of the induced subgraph with selected nodes.
    """
    if max_nodes is None or G.number_of_nodes() <= max_nodes:
        return G.copy()

    method = method.lower()
    if method == "pagerank":
        try:
            centrality = nx.pagerank(G)
        except Exception:
            centrality = dict(G.degree())
    elif method == "degree":
        centrality = nx.degree_centrality(G)
    elif method == "betweenness":
        centrality = nx.betweenness_centrality(G)
    else:
        raise ValueError(f"Unsupported centrality method: {method}")

    # Sort nodes by centrality descending
    sorted_nodes = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
    selected = [n for n, _ in sorted_nodes[:max_nodes]]
    selected_set = set(selected)

    # Optionally expand to include neighbors to preserve connectivity/context
    if expand_neighbors:
        for n, _ in sorted_nodes[: max_nodes * 2]:
            if len(selected_set) >= max_nodes:
                break
            for nbr in G.neighbors(n):
                if len(selected_set) >= max_nodes:
                    break
                selected_set.add(nbr)
        selected = list(selected_set)[:max_nodes]

    return G.subgraph(selected).copy()


if __name__ == "__main__":
    import json
    import os
    from pathlib import Path

    # env setup
    from dotenv import load_dotenv
    load_dotenv()

    from class_factory.concept_web.build_concept_map import (
        build_graph, detect_communities)

    # Path definitions
    projectDir = Path.home() / Path(os.getenv('projectDir'))
    dataDir = projectDir / "data/"

    with open(dataDir / 'interim/conceptlist_test.json', 'r') as f:
        concept_list = json.load(f)

    with open(dataDir / 'interim/relationship_list_test.json', 'r') as f:
        relationship_list = json.load(f)

    # Create and save the interactive graph as an HTML file
    output_path = str(dataDir / "interim/interactive_concept_map_test.html")

    # Build the graph
    G_base = build_graph(relationship_list)
    # Detect communities using Louvain method
    G = detect_communities(G_base, method="leiden")

    visualize_graph_interactive(G, output_path, dark_mode=False, max_nodes=500)
