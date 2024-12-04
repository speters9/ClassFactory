import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import PromptTemplate

from class_factory.utils.llm_validator import Validator
from class_factory.utils.tools import logger_setup


@pytest.fixture
def mock_llm():
    """Fixture to mock the language model."""
    mock_llm = MagicMock()
    mock_llm.temperature = 0.6
    return mock_llm


@pytest.fixture
def mock_parser():
    """Fixture to mock the JSON output parser."""
    mock_parser = MagicMock()
    return mock_parser


@pytest.fixture
def validator(mock_llm, mock_parser):
    """Fixture to create a Validator instance with mocked components."""
    with patch('langchain_core.prompts.PromptTemplate.from_template') as mock_template:
        mock_template.return_value = MagicMock()
        validator = Validator(llm=mock_llm, parser=mock_parser)

        def mock_chain_invoke(input_dict):
            return {
                "evaluation_score": 8.5,
                "status": 1,
                "reasoning": "Test reasoning",
                "additional_guidance": ""
            }

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = mock_chain_invoke
        mock_template.return_value.__or__.return_value.__or__.return_value = mock_chain

        return validator


def test_validator_initialization(mock_llm, mock_parser):
    """Test proper initialization of Validator class."""
    temperature = 0.4
    validator = Validator(llm=mock_llm, parser=mock_parser, temperature=temperature)

    assert validator.llm == mock_llm
    assert validator.parser == mock_parser
    assert validator.llm.temperature == temperature
    assert validator.logger.name == "validator"


def test_validator_successful_validation(validator):
    """Test successful validation with high evaluation score."""
    task_description = "Summarize the key points."
    generated_response = "This is a good summary."
    specific_guidance = "Focus on accuracy."

    expected_response = {
        "evaluation_score": 8.5,
        "status": 1,
        "reasoning": "The response is accurate and complete.",
        "additional_guidance": ""
    }

    # Mock the LLM chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = expected_response

    # Patch ChatPromptTemplate to ensure it produces the expected mock chain
    with patch('langchain.prompts.ChatPromptTemplate.from_messages') as mock_prompt_template:
        # Mock the chain to return our mock_chain
        mock_prompt_template.return_value.__or__.return_value.__or__.return_value = mock_chain
        result = validator.validate(task_description, generated_response, specific_guidance)

    assert isinstance(result, dict)
    assert result["evaluation_score"] == 8.5
    assert result["status"] == 1
    assert result["reasoning"]
    assert result["additional_guidance"] == ""


def test_validator_failed_validation(validator):
    """Test failed validation with low evaluation score."""
    task_description = "Summarize the document."
    generated_response = "Incomplete summary."
    specific_guidance = "Be thorough."

    expected_response = {
        "evaluation_score": 5.0,
        "status": 0,
        "reasoning": "The response is incomplete.",
        "additional_guidance": "Include all main points."
    }

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = expected_response

    # Patch ChatPromptTemplate to ensure it produces the expected mock chain
    with patch('langchain.prompts.ChatPromptTemplate.from_messages') as mock_prompt_template:
        # Mock the chain to return our mock_chain
        mock_prompt_template.return_value.__or__.return_value.__or__.return_value = mock_chain
        result = validator.validate(task_description, generated_response, specific_guidance)

    assert isinstance(result, dict)
    assert result["evaluation_score"] == 5.0
    assert result["status"] == 0
    assert result["reasoning"]
    assert result["additional_guidance"] == "Include all main points."


def test_validator_with_empty_specific_guidance(validator):
    """Test validation when no specific guidance is provided."""
    task_description = "Summarize the text."
    generated_response = "Basic summary."

    expected_response = {
        "evaluation_score": 7.0,
        "status": 1,
        "reasoning": "Adequate response.",
        "additional_guidance": ""
    }

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = expected_response

    # Patch ChatPromptTemplate to ensure it produces the expected mock chain
    with patch('langchain.prompts.ChatPromptTemplate.from_messages') as mock_prompt_template:
        # Mock the chain to return our mock_chain
        mock_prompt_template.return_value.__or__.return_value.__or__.return_value = mock_chain
        result = validator.validate(task_description, generated_response)

    assert isinstance(result, dict)
    assert "evaluation_score" in result
    assert "status" in result


def test_validator_prompt_template_format(validator):
    """Test that the prompt template contains all required components."""
    task_description = "Test task"
    generated_response = "Test response"

    with patch('langchain.prompts.ChatPromptTemplate.from_messages') as mock_prompt_template:
        validator.validate(task_description, generated_response)

        # Verify the template was called with a string containing required elements
        template_str = mock_prompt_template.call_args[0][0]
        assert "evaluation_score" in template_str[1].prompt.template
        assert "status" in template_str[1].prompt.template
        assert "reasoning" in template_str[1].prompt.template
        assert "additional_guidance" in template_str[1].prompt.template


def test_validator_retry_functionality(validator, caplog):
    """Test the retry decorator on validation method."""
    task_description = "Test task"
    generated_response = "Test response"

    # Mock chain to fail with JSON error first, then succeed
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [
        json.JSONDecodeError("Test error", "test", 0),  # First call fails
        {  # Second call succeeds
            "evaluation_score": 8.0,
            "status": 1,
            "reasoning": "Good response",
            "additional_guidance": ""
        }
    ]

    # Patch ChatPromptTemplate to ensure it produces the expected mock chain
    with patch('langchain.prompts.ChatPromptTemplate.from_messages') as mock_prompt_template:
        # Mock the chain to return our mock_chain
        mock_prompt_template.return_value.__or__.return_value.__or__.return_value = mock_chain
        result = validator.validate(task_description, generated_response)

    assert isinstance(result, dict)
    assert result["status"] == 1
    assert any('error' in record.message.lower() for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__])
