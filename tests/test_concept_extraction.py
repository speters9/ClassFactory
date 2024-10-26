import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from class_factory.concept_web.concept_extraction import (
    extract_concepts_from_relationships, extract_relationships,
    jaccard_similarity, normalize_concept, process_relationships,
    replace_similar_concepts, summarize_text)

logging.getLogger('httpx').setLevel(logging.WARNING)


def test_extract_relationships_error_handling(caplog):
    # Mock the llm to return an invalid JSON response
    mock_llm = MagicMock()
    mock_llm_response = "Invalid JSON response"

    # Mock the chain object
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_llm_response

    # Mock the combined_template object
    mock_combined_template = MagicMock()

    # Mock the intermediate chain
    mock_after_llm = MagicMock()

    # Set up the chain of __or__ methods
    mock_combined_template.__or__.return_value = mock_after_llm
    mock_after_llm.__or__.return_value = mock_chain

    # Patch PromptTemplate.from_template
    with patch('class_factory.concept_web.concept_extraction.PromptTemplate') as mock_prompt_template_class:
        mock_prompt_template_class.from_template.return_value = mock_combined_template

        # Patch JsonOutputParser
        with patch('class_factory.concept_web.concept_extraction.JsonOutputParser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser

            # Patch time.sleep to avoid delays during testing
            with patch('time.sleep', return_value=None) as mock_sleep:
                with caplog.at_level(logging.ERROR):
                    max_retries = 3

                    # Simulate JSONDecodeError being raised
                    with patch('json.loads', side_effect=json.JSONDecodeError('Invalid json output:', '', 0)):
                        # This is where we expect the exception to be raised
                        with pytest.raises(json.JSONDecodeError):
                            extract_relationships(
                                text="Some text",
                                objectives="Some objectives",
                                course_name="Test Course",
                                llm=mock_llm,
                                verbose=False
                            )

                # Check that the function retried the correct number of times
                assert mock_chain.invoke.call_count == max_retries
                assert mock_sleep.call_count == max_retries - 1

                # Verify that the error was logged the correct number of times
                error_logs = [record for record in caplog.records if record.levelno == logging.ERROR]
                assert len(error_logs) == max_retries + 1  # Including the final log before raising

                # Optionally, check the content of the last log message
                assert "Max retries reached. Raising the exception." in error_logs[-1].message


def test_summarize_text():
    # Mock the LLM to return a predefined summary
    mock_llm = MagicMock()
    mock_summary = "This is a summary of the text."

    # Mock the chain object that will be the result of the chain 'summary_template | llm | parser'
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_summary

    # Mock the summary_template object
    mock_summary_template = MagicMock()

    # Mock the intermediate chain after summary_template | llm
    mock_after_llm = MagicMock()

    # Set up the chain of __or__ methods
    # summary_template | llm returns mock_after_llm
    mock_summary_template.__or__.return_value = mock_after_llm

    # mock_after_llm | parser returns mock_chain
    mock_after_llm.__or__.return_value = mock_chain

    # Patch PromptTemplate.from_template to return the mock_summary_template
    with patch('class_factory.concept_web.concept_extraction.PromptTemplate') as mock_prompt_template_class:
        mock_prompt_template_class.from_template.return_value = mock_summary_template

        # Patch StrOutputParser if necessary
        with patch('class_factory.concept_web.concept_extraction.StrOutputParser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser

            # Call the function
            result = summarize_text(
                text="Some text",
                prompt="Summarize the following text:",
                course_name="Test Course",
                llm=mock_llm,
                verbose=False
            )

            # Verify that the function returns the expected summary
            assert result == mock_summary

            # Verify that the chain was invoked with correct arguments
            mock_chain.invoke.assert_called_once_with({
                'course_name': "Test Course",
                'text': "Some text"
            })


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
