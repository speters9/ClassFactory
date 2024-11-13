import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from class_factory.concept_web.ConceptWeb import ConceptMapBuilder


@pytest.fixture
def mock_paths():
    project_dir = Path("/mocked/path/project")
    readings_dir = Path("/mocked/path/readings")
    syllabus_path = Path("/mocked/path/syllabus.docx")
    output_dir = Path("/mocked/path/output")

    return {
        "project_dir": project_dir,
        "readings_dir": readings_dir,
        "syllabus_path": syllabus_path,
        "output_dir": output_dir
    }


@pytest.fixture
def mock_llm():
    return Mock()

# Fixture to provide a reusable instance of ConceptMapBuilder


@pytest.fixture
def builder(mock_llm, mock_paths):
    with patch('class_factory.concept_web.ConceptWeb.ConceptMapBuilder._validate_file_path', return_value=mock_paths["syllabus_path"]), \
            patch('class_factory.concept_web.ConceptWeb.ConceptMapBuilder._validate_dir_path', side_effect=lambda path, name: path), \
            patch('pathlib.Path.is_file', return_value=True):
        return ConceptMapBuilder(
            project_dir=mock_paths["project_dir"],
            readings_dir=mock_paths["readings_dir"],
            syllabus_path=mock_paths["syllabus_path"],
            llm=mock_llm,
            course_name="Test Course",
            lesson_range=range(1, 3),
            output_dir=mock_paths["output_dir"]
        )

# Test the __init__ method


def test_concept_map_builder_init(builder, mock_paths):
    """Test ConceptMapBuilder initialization with the fixture."""
    assert builder.project_dir == mock_paths["project_dir"]
    assert builder.readings_dir == mock_paths["readings_dir"]
    assert builder.syllabus_path == mock_paths["syllabus_path"]
    assert builder.output_dir == mock_paths["output_dir"] / f"L{min(builder.lesson_range)}_{max(builder.lesson_range)}"
    assert isinstance(builder.llm, Mock)
    assert builder.course_name == "Test Course"

# Test setting user objectives with a valid list


def test_set_user_objectives_with_valid_list(builder):
    objectives = ['Objective 1', 'Objective 2']
    builder._set_user_objectives(objectives)
    expected = {'Lesson 1': 'Objective 1', 'Lesson 2': 'Objective 2'}
    assert builder.user_objectives == expected

# Test setting user objectives with an invalid length


def test_set_user_objectives_with_invalid_length(builder):
    objectives = ['Objective 1']
    with pytest.raises(ValueError, match="Length of objectives list must match the number of lessons"):
        builder._set_user_objectives(objectives)

# Test setting user objectives with an invalid type


def test_set_user_objectives_invalid_type(builder):
    objectives = 'Invalid Type'
    with pytest.raises(TypeError, match="Objectives must be provided as either a list or a dictionary."):
        builder._set_user_objectives(objectives)

# Mock the lesson processing and summarize/extract methods


@patch('class_factory.concept_web.ConceptWeb.load_lessons')
@patch('class_factory.concept_web.ConceptWeb.extract_lesson_objectives')
@patch('class_factory.concept_web.ConceptWeb.summarize_text')
@patch('class_factory.concept_web.ConceptWeb.extract_relationships')
@patch('class_factory.concept_web.ConceptWeb.extract_concepts_from_relationships')
@patch('class_factory.concept_web.ConceptWeb.process_relationships')
def test_load_and_process_lessons(
    mock_process_relationships,
    mock_extract_concepts,
    mock_extract_relationships,
    mock_summarize_text,
    mock_extract_lesson_objectives,
    mock_load_lessons,
    builder
):
    # Arrange the mock returns
    mock_load_lessons.return_value = ['Document 1', 'Document 2']
    mock_extract_lesson_objectives.return_value = 'Lesson Objectives'
    mock_summarize_text.return_value = 'Summary Text'
    mock_extract_relationships.return_value = [('Concept A', 'relates to', 'Concept B')]
    mock_extract_concepts.return_value = ['Concept A', 'Concept B']
    mock_process_relationships.return_value = [('Concept A', 'relates to', 'Concept B')]

    # Act
    builder.load_and_process_lessons(
        summary_prompt='Summary Prompt',
        relationship_prompt='Relationship Prompt'
    )

    # Assert that the methods were called correctly
    assert mock_extract_lesson_objectives.call_count == 2
    assert mock_load_lessons.call_count == 2
    assert mock_summarize_text.call_count == 4  # two documents, two directories
    mock_summarize_text.assert_any_call('Document 1', prompt='Summary Prompt', course_name='Test Course', llm=builder.llm)
    mock_summarize_text.assert_any_call('Document 2', prompt='Summary Prompt', course_name='Test Course', llm=builder.llm)

    # Check that the relationship_list has the expected relationships
    assert len(builder.relationship_list) == 1  # It should contain two relationships
    assert all(rel == ('Concept A', 'relates to', 'Concept B') for rel in builder.relationship_list)


# Mock the graph-building and visualization functions
@patch('class_factory.concept_web.ConceptWeb.visualize_graph_interactive')
@patch('class_factory.concept_web.ConceptWeb.generate_wordcloud')
@patch('class_factory.concept_web.ConceptWeb.build_graph')
@patch('class_factory.concept_web.ConceptWeb.detect_communities')
def test_build_and_visualize_graph(
    mock_detect_communities,
    mock_build_graph,
    mock_generate_wordcloud,
    mock_visualize_graph_interactive,
    builder
):
    # Arrange the mock returns
    builder.relationship_list = [('Concept A', 'relates to', 'Concept B')]
    mock_build_graph.return_value = MagicMock()

    # Act
    builder._build_and_visualize_graph()

    # Assert that the methods were called correctly
    mock_build_graph.assert_called_once_with(
        relationships=builder.relationship_list,
        directed=False,
        concept_similarity_threshold=0.85
    )
    mock_detect_communities.assert_called_once()  # two lessons covered
    mock_visualize_graph_interactive.assert_called_once()
    mock_generate_wordcloud.assert_called_once()

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


def test_output_dir_creation(tmp_path):
    # Create temporary directories for the test
    project_dir = tmp_path / "project"
    readings_dir = tmp_path / "readings"
    syllabus_path = tmp_path / "syllabus.docx"
    output_dir = tmp_path / "output"

    project_dir.mkdir()
    readings_dir.mkdir()
    syllabus_path.touch()

    builder = ConceptMapBuilder(
        project_dir=project_dir,
        readings_dir=readings_dir,
        syllabus_path=syllabus_path,
        llm=MagicMock(),
        course_name='Test Course',
        lesson_range=range(1, 2),
        output_dir=output_dir
    )

    # Assert that the output directory exists
    assert builder.output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__])
