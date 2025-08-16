import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from class_factory.concept_web.ConceptWeb import ConceptMapBuilder
from class_factory.utils.load_documents import LessonLoader


@pytest.fixture
def mock_llm():
    return Mock()

# Fixture to provide a reusable instance of ConceptMapBuilder


@pytest.fixture
def mock_paths():
    # Create temporary directories
    temp_dirs = {
        "syllabus_path": Path(tempfile.mkdtemp()) / "syllabus.txt",
        "reading_dir": Path(tempfile.mkdtemp()),
        "slide_dir": Path(tempfile.mkdtemp()),
        "output_dir": Path(tempfile.mkdtemp()),
        "project_dir": Path(tempfile.mkdtemp()),
    }
    # Create a dummy syllabus file
    temp_dirs["syllabus_path"].write_text("Syllabus content here.")

    # Yield the paths to the test
    yield temp_dirs

    # Cleanup after the test
    for temp_dir in temp_dirs.values():
        if temp_dir.is_dir():
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_lesson_loader(mock_paths):
    # Create mock readings within the reading directory
    lesson_dir = mock_paths["reading_dir"] / "L1"
    lesson_dir.mkdir(parents=True)
    (lesson_dir / "reading1.txt").write_text("Reading 1 content")
    (lesson_dir / "reading2.txt").write_text("Reading 2 content")

    return LessonLoader(
        syllabus_path=mock_paths["syllabus_path"],
        reading_dir=mock_paths["reading_dir"],
        slide_dir=mock_paths["slide_dir"],
        project_dir=mock_paths["project_dir"]
    )


@pytest.fixture
def builder(mock_llm, mock_paths, mock_lesson_loader):
    with patch('pathlib.Path.is_file', return_value=True):
        return ConceptMapBuilder(
            lesson_loader=mock_lesson_loader,
            llm=mock_llm,
            lesson_no=1,
            course_name="Test Course",
            lesson_range=range(1, 3),
            output_dir=mock_paths["output_dir"]
        )

# Test the __init__ method


def test_concept_map_builder_init(builder, mock_paths):
    """Test ConceptMapBuilder initialization with the fixture."""
    assert builder.lesson_loader.project_dir == mock_paths["project_dir"]
    assert builder.lesson_loader.reading_dir == mock_paths["reading_dir"]
    assert builder.lesson_loader.syllabus_path == mock_paths["syllabus_path"]
    assert builder.output_dir == mock_paths["output_dir"] / f"L{min(builder.lesson_range)}_{max(builder.lesson_range)}"
    # Assert that the output directory exists
    assert builder.output_dir.exists()
    assert isinstance(builder.llm, Mock)
    assert builder.course_name == "Test Course"

# Test setting user objectives with a valid list


def test_set_user_objectives_with_valid_list(builder):
    objectives = ['Objective 1', 'Objective 2']
    objs = builder.set_user_objectives(objectives, lesson_range=range(1, 3))
    builder.user_objectives = objs
    expected = {'1': 'Objective 1', '2': 'Objective 2'}
    assert builder.user_objectives == expected

# Test setting user objectives with an invalid length


def test_set_user_objectives_with_invalid_length(builder):
    objectives = ['Objective 1']
    with pytest.raises(ValueError, match="Length of objectives list must match the number of lessons"):
        builder.set_user_objectives(objectives, lesson_range=range(1, 3))

# Test setting user objectives with an invalid type


def test_set_user_objectives_invalid_type(builder):
    objectives = 'Invalid Type'
    with pytest.raises(TypeError, match="Objectives must be provided as either a list or a dictionary."):
        builder.set_user_objectives(objectives, lesson_range=range(1, 2))

# Mock the lesson processing and summarize/extract methods


@patch('class_factory.concept_web.ConceptWeb.summarize_text')
@patch('class_factory.concept_web.ConceptWeb.extract_relationships')
@patch('class_factory.concept_web.ConceptWeb.extract_concepts_from_relationships')
@patch('class_factory.concept_web.ConceptWeb.process_relationships')
def test_load_and_process_lessons(
    mock_process_relationships,
    mock_extract_concepts,
    mock_extract_relationships,
    mock_summarize_text,
    builder
):
    # Setup instance variables
    builder.lesson_range = range(1, 3)  # Lessons 1 and 2
    builder.readings = {
        '1': ['Document 1'],
        '2': ['Document 2'],
        '3': ['Document 3']  # This one should be skipped
    }
    builder.user_objectives = {}

    # Arrange mock returns
    mock_extract_lesson_objectives = MagicMock(return_value='Lesson Objectives')
    mock_summarize_text.return_value = 'Summary Text'
    mock_extract_relationships.return_value = [('Concept A', 'relates to', 'Concept B')]
    mock_extract_concepts.return_value = ['Concept A', 'Concept B']
    mock_process_relationships.return_value = [('Concept A', 'relates to', 'Concept B')]

    # Act
    with patch.object(builder.lesson_loader, "extract_lesson_objectives", mock_extract_lesson_objectives):
        builder.load_and_process_lessons()

        # Verify only lessons 1 and 2 were processed
        assert mock_extract_lesson_objectives.call_count == 2
        mock_extract_lesson_objectives.assert_has_calls([
            call(1, only_current=True),
            call(2, only_current=True)
        ])

        # Verify correct number of document processing calls
        assert mock_summarize_text.call_count == 2  # One for each document in lessons 1 and 2
        assert mock_extract_relationships.call_count == 2
        assert mock_extract_concepts.call_count == 2

        # Verify final processing
        assert mock_process_relationships.call_count == 1
        assert len(builder.relationship_list) == 1


# Mock the graph-building and visualization functions
@patch('class_factory.concept_web.ConceptWeb.visualize_graph_interactive')
@patch('class_factory.concept_web.ConceptWeb.build_graph')
@patch('class_factory.concept_web.ConceptWeb.detect_communities')
def test_build_and_visualize_graph(
    mock_detect_communities,
    mock_build_graph,
    mock_visualize_graph_interactive,
    builder
):
    # Arrange the mock returns
    builder.relationship_list = [('Concept A', 'relates to', 'Concept B')]
    from unittest.mock import patch as unittest_patch

    import networkx as nx

    # Create a valid NetworkX graph with string node names and required attributes
    mock_graph = nx.Graph()
    mock_graph.add_edge('Concept A', 'Concept B', relation={'relates to'}, weight=1)
    # Set required node attributes
    nx.set_node_attributes(mock_graph, {
        'Concept A': {'text_size': 10, 'community': 0},
        'Concept B': {'text_size': 10, 'community': 0}
    })
    mock_build_graph.return_value = mock_graph
    # Mock detect_communities with a simpler implementation
    mock_detect_communities.side_effect = lambda G, method='leiden': G

    # Set up builder for two lessons to ensure community detection runs
    builder.lesson_range = range(1, 3)

    # Patch bipartite.is_bipartite to always return False for this test
    with unittest_patch('networkx.algorithms.bipartite.is_bipartite', return_value=False):
        # Act
        builder._build_and_visualize_graph()

    # Assert that the methods were called correctly
    mock_build_graph.assert_called_once_with(
        processed_relationships=builder.relationship_list,
        directed=False
    )
    # Check detect_communities was called with the correct graph and method
    mock_detect_communities.assert_called_once_with(mock_graph, method='leiden')
    mock_visualize_graph_interactive.assert_called_once()

# Mock the build_concept_map steps


@patch.object(ConceptMapBuilder, '_build_and_visualize_graph')
@patch.object(ConceptMapBuilder, 'load_and_process_lessons')
def test_build_concept_map(mock_load_and_process_lessons, mock_build_and_visualize_graph, builder):
    # Act
    builder.build_concept_map()

    # Assert that the methods were called in the right order
    mock_load_and_process_lessons.assert_called_once()
    mock_build_and_visualize_graph.assert_called_once()

# Test if output directory creation works


if __name__ == "__main__":
    pytest.main([__file__])
