import logging
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from class_factory.concept_web.build_concept_map import (build_graph,
                                                         detect_communities)


def test_build_graph(caplog):
    # Example relationships: (concept1, relationship, concept2)
    relationships = [("ConceptA", "relates_to", "ConceptB"),
                     ("ConceptB", "relates_to", "ConceptC"),
                     ("ConceptA", "None", "ConceptC")]

    # Capture logs at the WARNING level
    with caplog.at_level(logging.WARNING):
        G = build_graph(relationships)

        # Test if the graph has the correct nodes and edges
        assert len(G.nodes) == 3  # ConceptA, ConceptB, ConceptC
        assert len(G.edges) == 2  # Only ConceptA <-> ConceptB and ConceptB <-> ConceptC

        # Check if the edge weights are normalized correctly, or if normalization was skipped
        if any("normalized_weight" in G[u][v] for u, v in G.edges):
            # If normalization happened, check for normalized weights
            assert G["ConceptA"]["ConceptB"]["normalized_weight"] is not None
        else:
            # If no variation, normalized_weight should not exist, and we expect the warning
            assert "normalized_weight" not in G["ConceptA"]["ConceptB"]

            # Check that the expected warning was logged
            assert "Normalization of weights and centrality skipped due to lack of variation in the graph." in caplog.text


def test_build_graph_zero_division(caplog):
    # Create relationships with no variation (all identical)
    relationships = [("ConceptA", "relates_to", "ConceptB")]

    # Capture logs at the WARNING level
    with caplog.at_level(logging.WARNING):
        G = build_graph(relationships, concept_similarity_threshold=0.9)

        # Ensure that fallback sizes are applied due to ZeroDivisionError
        for node in G.nodes:
            assert G.nodes[node]['text_size'] == 12
            assert G.nodes[node]['centrality'] == 0.5

        # Check that the expected warning was logged
        assert "Normalization of weights and centrality skipped due to lack of variation in the graph." in caplog.text


def test_build_graph_zero_division():
    # Create relationships with no variation (all identical)
    relationships = [("ConceptA", "relates_to", "ConceptB")]

    G = build_graph(relationships)

    # Ensure that fallback sizes are applied due to ZeroDivisionError
    for node in G.nodes:
        assert G.nodes[node]['text_size'] == 12
        assert G.nodes[node]['centrality'] == 0.5

# Mocking the cdlib library and its 'leiden' algorithm


def test_detect_communities_leiden():
    # Create a simple graph
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

    # Mock the cdlib algorithms.leiden
    with patch('cdlib.algorithms.leiden') as mock_leiden:
        # Simulate the return value of the leiden algorithm (list of communities)
        mock_leiden.return_value.communities = [{"A", "B"}, {"C", "D"}]

        G_with_communities = detect_communities(G, method="leiden")

        # Verify that community labels are assigned
        assert G_with_communities.nodes["A"]["community"] == 0
        assert G_with_communities.nodes["C"]["community"] == 1

# Mocking the networkx louvain community detection


def test_detect_communities_louvain():
    # Create a simple graph
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

    # Mock the nx_comm.louvain_communities
    with patch('networkx.algorithms.community.louvain_communities') as mock_louvain:
        # Simulate the return value of the louvain algorithm (list of communities)
        mock_louvain.return_value = [{"A", "B"}, {"C", "D"}]

        G_with_communities = detect_communities(G, method="louvain")

        # Verify that community labels are assigned
        assert G_with_communities.nodes["A"]["community"] == 0
        assert G_with_communities.nodes["C"]["community"] == 1


def test_detect_communities_invalid_method():
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C")])

    # Ensure that using an invalid method raises a ValueError
    with pytest.raises(ValueError, match="Unknown method"):
        detect_communities(G, method="invalid_method")

# Mocking the sklearn SpectralClustering


def test_detect_communities_spectral():
    # Create a small graph
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

    # Patch the SpectralClustering to control the output
    with patch('sklearn.cluster.SpectralClustering') as mock_spectral:
        # Mock the return value of fit_predict to simulate clustering
        mock_spectral.return_value.fit_predict.return_value = [0, 0, 1, 1]

        G_with_communities = detect_communities(G, method="spectral", num_clusters=2)

        # Verify that community labels are assigned based on the mock clusters
        assert (G_with_communities.nodes["A"]["community"] == 0 and
                G_with_communities.nodes["D"]["community"] == 1) | (G_with_communities.nodes["A"]["community"] == 1 and
                                                                    G_with_communities.nodes["D"]["community"] == 0)


if __name__ == "__main__":
    pytest.main([__file__])
