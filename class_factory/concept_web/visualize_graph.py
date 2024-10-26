
from pathlib import Path
from typing import List, Union

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

- `generate_wordcloud(concept_list: List[str], output_path: str = None) -> None`:
    Generates a word cloud image from a list of concepts, optionally saving the result to a file. The word cloud
    visually represents the frequency of concepts, with more frequent concepts displayed more prominently.

Workflow:
1. **Graph Conversion**: Converts the provided NetworkX graph into a pyvis graph, applying styles and attributes
   like node size and edge width based on centrality and relationship frequency.
2. **Interactive Visualization**: Saves the interactive graph as an HTML file, which can be explored in any web browser.
3. **Word Cloud Creation**: Generates a word cloud image from the list of concepts and optionally saves it to disk.

Dependencies:
- NetworkX: For graph data structure and manipulation.
- Matplotlib: For color mapping and displaying the word cloud.
- Pyvis: For creating interactive graph visualizations in HTML.
- WordCloud: For generating word cloud images.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
from wordcloud import WordCloud


def visualize_graph_interactive(G: nx.Graph, output_path: Union[Path, str], directed: bool = False) -> None:
    """
    Visualizes the graph interactively using pyvis and saves it as an HTML file.
    Includes options for physics simulations, node filtering, and clustering.

    Args:
        G (networkx.Graph): The graph to visualize.
        output_path (Union[Path, str]): The file path where the HTML file will be saved.
        directed (bool): If True, adds arrows to edges to represent directionality.
    """
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', directed=directed)

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
        node["size"] = node['text_size']
        node["font"].update({"size": node['text_size']})

    for edge in net.edges:
        edge['relation'] = list(edge['relation'])
        edge['title'] = ", ".join(edge['relation'])
        edge['width'] = edge.get('normalized_weight', 'weight')
        if directed:
            edge["arrows"] = "to"

    # Add physics controls for a dynamic layout
    net.show_buttons(filter_=['layout'])  # ['physics'])

    output_path = str(output_path)
    # Save the network as an HTML file
    net.save_graph(output_path)
    print(f"Concept map saved to {output_path}")

    # Optionally, you can also open it directly in a browser
    # net.show(output_path)


def generate_wordcloud(concept_list: List[str], output_path: str = None) -> None:
    """
    Generates and optionally saves a word cloud image from a list of concepts.

    Args:
        concept_list (List[str]): The list of concepts to visualize in the word cloud.
        output_path (str, optional): The file path to save the word cloud image. If None, the word cloud is only displayed.
    """
    # Create a string with each concept repeated according to its frequency
    concept_string = " ".join(concept_list)

    # Generate the word cloud
    wordcloud = WordCloud(width=1500, height=1000, background_color='white', max_font_size=150, max_words=250).generate(concept_string)

    # Display the word cloud
    # plt.figure(figsize=(10, 5))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()

    if output_path:
        wordcloud.to_file(output_path)


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
    projectDir = Path(os.getenv('projectDir'))
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

    visualize_graph_interactive(G, output_path)
    generate_wordcloud(concept_list)
