import json
import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from class_factory.concept_web.concept_extraction import (
    extract_concepts_from_relationships, extract_relationships,
    normalize_for_embedding, normalize_for_output, process_relationships,
    replace_similar_concepts, summarize_text)

logging.getLogger('httpx').setLevel(logging.WARNING)


@pytest.fixture
def mock_llm():
    """Fixture to mock the language model."""
    mock_llm = MagicMock()
    mock_llm.temperature = 0.6
    return mock_llm


@pytest.fixture
def basic_prompt():
    return ChatPromptTemplate.from_template("""
    Summarize the content for the course {course_name}.
    Content: {text}
    """)


def test_summarize_text(mock_llm, basic_prompt):
    # Configure mock response
    mock_llm.return_value = "Mocked summary response"

    result = summarize_text(
        text="Test content",
        prompt=basic_prompt,
        course_name="Test Course",
        llm=mock_llm,
        parser=StrOutputParser()
    )

    # Verify the result
    assert isinstance(result, str)
    assert result == "Mocked summary response"

    # Verify the mock was called with correct arguments
    mock_llm.assert_called_once()


def test_extract_relationships_error_handling(caplog):
    # Mock the LLM to consistently return an invalid JSON response
    mock_llm = MagicMock()
    mock_llm.side_effect = json.JSONDecodeError('Invalid json output:', '', 0)

    # Set up logging
    max_retries = 3
    with caplog.at_level(logging.ERROR):
        # Expect the final JSONDecodeError to be raised
        # Patch time.sleep to avoid delays during testing
        with patch('time.sleep', return_value=None) as mock_sleep:
            with pytest.raises(json.JSONDecodeError):
                extract_relationships(
                    text="Some text",
                    objectives="Some objectives",
                    course_name="Test Course",
                    llm=mock_llm,
                    verbose=False
                )

    # Check retry logs
    error_logs = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(error_logs) == max_retries + 1  # Including final error before raising
    assert error_logs[-1].message == "Max retries reached. Raising the exception."


def test_extract_concepts_from_relationships():
    relationships = [
        ("Concept1", "related_to", "Concept2"),
        ("Concept2", "leads_to", "Concept3"),
        ("Concept1", "associated_with", "Concept3")
    ]
    expected_concepts = ["concept1", "concept2", "concept3"]  # should be made lower in the process

    result = extract_concepts_from_relationships(relationships)

    # Since the order may vary, compare sets
    assert set(result) == set(expected_concepts)


def test_normalize_for_output():
    """Test the normalize_for_output function."""
    test_cases = [
        ("Human Nature", "Human_Nature"),
        ("States of War", "States_of_War"),
        ("  Social  Contracts  ", "Social_Contracts"),
        ("is this valid", "this_valid"),  # Removes "is"
        ("is", ""),  # Single "is" becomes an empty string
    ]

    for input_concept, expected_output in test_cases:
        result = normalize_for_output(input_concept)
        assert result == expected_output, f"Failed for input: {input_concept}"


def test_normalize_for_embedding():
    """Test the normalize_for_embedding function."""
    # Test single string input
    test_cases_single = [
        ("Human Nature", "human nature"),
        ("States of War", "state of war"),
        ("  Social  Contracts  ", "social contract"),
    ]

    for input_concept, expected_output in test_cases_single:
        result = normalize_for_embedding(input_concept)
        assert result == expected_output, f"Failed for input: {input_concept}"

    # Test list input
    test_cases_list = [
        (["Human Nature", "States of War"], ["human nature", "state of war"]),
        (["  Social  Contracts  ", "Orders_and Securities"], ["social contract", "order and security"]),
    ]

    for input_concepts, expected_outputs in test_cases_list:
        result = normalize_for_embedding(input_concepts)
        assert result == expected_outputs, f"Failed for input: {input_concepts}"


def test_replace_similar_concepts():
    # Mock embeddings for testing
    concept_embeddings = {
        "concept1": torch.tensor([1.0, 0.0, 0.0]),
        "concept2": torch.tensor([0.0, 1.0, 0.0]),
        "concept3": torch.tensor([0.0, 0.0, 1.0]),
        "similar_to_concept1": torch.tensor([0.995, 0.01, 0.0]),  # Close to "concept1"
    }

    existing_concepts = {"concept1", "concept2"}

    # Case 1: New concept is similar to an existing one
    new_concept_similar = "similar_to_concept1"
    result_similar = replace_similar_concepts(existing_concepts, new_concept_similar, concept_embeddings, threshold=0.995)
    assert result_similar == "concept1", f"Failed for similar concept: {new_concept_similar}"

    # Case 2: New concept is different
    new_concept_different = "concept3"
    result_different = replace_similar_concepts(existing_concepts, new_concept_different, concept_embeddings, threshold=0.995)
    assert result_different == "concept3", f"Failed for different concept: {new_concept_different}"

    # Case 3: Similarity below the threshold
    concept_embeddings["barely_similar"] = torch.tensor([0.8, 0.2, 0.0])  # Not quite similar enough
    new_concept_barely_similar = "barely_similar"
    result_barely_similar = replace_similar_concepts(existing_concepts, new_concept_barely_similar, concept_embeddings, threshold=0.995)
    assert result_barely_similar == "barely_similar", f"Failed for barely similar concept: {new_concept_barely_similar}"

    # Case 4: Empty existing_concepts set
    empty_existing_concepts = set()
    result_empty = replace_similar_concepts(empty_existing_concepts, new_concept_similar, concept_embeddings, threshold=0.995)
    assert result_empty == new_concept_similar, f"Failed for empty existing concepts with input: {new_concept_similar}"


def test_process_relationships():
    # Mocked dependencies
    def mock_extract_concepts_from_relationships(relationships):
        """Extract all unique concepts from relationships."""
        concepts = set()
        for c1, _, c2 in relationships:
            concepts.update([c1, c2])
        return list(concepts)

    def mock_normalize_for_embedding(concepts):
        """Normalize concepts for embedding."""
        if isinstance(concepts, str):
            return concepts.lower()
        return [concept.lower() for concept in concepts]

    def mock_get_embeddings(concepts):
        """Generate mock embeddings for testing."""
        embeddings = {}
        for concept in concepts:
            # Create simple mock embeddings: each concept's embedding is its ASCII sum normalized
            embeddings[concept] = torch.tensor([sum(ord(char) for char in concept) % 100 / 100.0])
        return embeddings

    # Monkey-patch the dependencies
    global extract_concepts_from_relationships, normalize_for_embedding, get_embeddings
    extract_concepts_from_relationships = mock_extract_concepts_from_relationships
    normalize_for_embedding = mock_normalize_for_embedding
    get_embeddings = mock_get_embeddings

    # Test relationships
    relationships = [
        ("Human Natures", "leads_to", "States of War"),
        ("human nature", "influences", "Social Contract"),
        ("State of War", "necessitates", "Commonwealths"),
        ("Commonwealth", "provides", "Order and Securities"),
    ]

    # Expected results after normalization and similarity replacement
    expected_processed_relationships = [
        ("human_nature", "leads_to", "state_of_war"),
        ("human_nature", "influences", "social_contract"),
        ("state_of_war", "necessitates", "commonwealth"),
        ("commonwealth", "provides", "order_and_security"),
    ]

    # Call the function under test
    result = process_relationships(relationships)

    # Validate the results
    assert result == expected_processed_relationships, f"Expected: {expected_processed_relationships}, Got: {result}"


if __name__ == "__main__":
    pytest.main([__file__])
