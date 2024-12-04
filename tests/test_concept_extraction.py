import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from class_factory.concept_web.concept_extraction import (
    extract_concepts_from_relationships, extract_relationships,
    jaccard_similarity, normalize_concept, process_relationships,
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
    expected_concepts = ["Concept1", "Concept2", "Concept3"]

    result = extract_concepts_from_relationships(relationships)

    # Since the order may vary, compare sets
    assert set(result) == set(expected_concepts)


def test_normalize_concept():
    test_cases = [
        ("Human Nature", "human_nature"),
        ("States of War", "state_of_war"),
        ("  Social  Contracts  ", "social_contract"),
        ("Sovereign Authorities", "sovereign_authority"),
        ("Orders_and Securities", "order_and_security"),
        (" IS ", "is")
    ]

    for input_concept, expected_output in test_cases:
        result = normalize_concept(input_concept)
        assert result == expected_output


def test_jaccard_similarity():
    test_cases = [
        ("concept", "concept", 0.85, True),
        ("concept", "concpt", 0.70, True),  # Missing 'e' but still similar
        ("concept", "different", 0.85, False),
        ("abcd", "abce", 0.75, False),  # similar, but intersection/union (3/5) is small due to small length
        ("abcd", "wxyz", 0.75, False)
    ]

    for c1, c2, threshold, expected in test_cases:
        result = jaccard_similarity(c1, c2, threshold)
        print(f'{c1}, {c2}')
        assert result == expected


def test_replace_similar_concepts():
    existing_concepts = {"concept1", "concept2"}
    new_concept_similar = "concept1"
    new_concept_different = "concept3"

    # When new concept is similar to an existing one
    result_similar = replace_similar_concepts(existing_concepts, new_concept_similar)
    assert result_similar == "concept1"

    # When new concept is different
    result_different = replace_similar_concepts(existing_concepts, new_concept_different)
    assert result_different == "concept3"


def test_process_relationships():
    relationships = [
        ("Human Natures", "leads_to", "States of War"),
        ("human nature", "influences", "Social Contract"),
        ("State of War", "necessitates", "Commonwealths"),
        ("Commonwealth", "provides", "Order and Securities")
    ]
    expected_processed_relationships = [
        ("human_nature", "lead_to", "state_of_war"),
        ("human_nature", "influence", "social_contract"),
        ("state_of_war", "necessitate", "commonwealth"),
        ("commonwealth", "provide", "order_and_security")
    ]

    result = process_relationships(relationships)

    assert result == expected_processed_relationships


if __name__ == "__main__":
    pytest.main([__file__])
