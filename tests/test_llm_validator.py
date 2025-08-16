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
def validator(mock_llm):
    """Fixture to create a Validator instance with mocked components."""
    return Validator(llm=mock_llm)


def test_validator_initialization(mock_llm):
    """Test proper initialization of Validator class."""
    temperature = 0.4
    validator = Validator(llm=mock_llm, temperature=temperature)

    assert validator.llm == mock_llm
    assert validator.llm.temperature == temperature
    assert validator.logger.name == "validator"


def test_validator_successful_validation(validator):
    """Test successful validation with high evaluation score."""
    task_description = "Summarize the key points."
    generated_response = "This is a good summary."
    specific_guidance = "Focus on accuracy."

    from class_factory.utils.llm_validator import ValidatorInterimResponse

    # Create a real ValidatorInterimResponse object with numeric values
    expected_response = {
        "accuracy": 8.5,
        "completeness": 8.5,
        "consistency": 8.5,
        "reasoning": "The response is accurate and complete.",
        "additional_guidance": ""
    }

    # Mock chain that returns a real ValidatorInterimResponse
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = ValidatorInterimResponse(**expected_response)
    with patch('class_factory.utils.llm_validator.ChatPromptTemplate.from_messages') as mock_template:
        # prompt | llm -> returns mock_chain
        mock_template.return_value.__or__.return_value = mock_chain
        result = validator.validate(task_description, generated_response, specific_guidance)

    assert isinstance(result, dict)
    assert result["overall_score"] == 8.5
    assert result["status"] == 1
    assert result["reasoning"]
    assert result["additional_guidance"] == ""


def test_validator_failed_validation(validator):
    """Test failed validation with low evaluation score."""
    task_description = "Summarize the document."
    generated_response = "Incomplete summary."
    specific_guidance = "Be thorough."

    from class_factory.utils.llm_validator import ValidatorInterimResponse

    # Create a real ValidatorInterimResponse object with numeric values
    expected_response = {
        "accuracy": 5.0,
        "completeness": 5.0,
        "consistency": 5.0,
        "reasoning": "The response is incomplete.",
        "additional_guidance": "Include all main points."
    }

    # Mock chain that returns a real ValidatorInterimResponse
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = ValidatorInterimResponse(**expected_response)
    with patch('class_factory.utils.llm_validator.ChatPromptTemplate.from_messages') as mock_template:
        mock_template.return_value.__or__.return_value = mock_chain
        result = validator.validate(task_description, generated_response, specific_guidance)

    assert isinstance(result, dict)
    assert result["overall_score"] == 5.0
    assert result["status"] == 0
    assert result["reasoning"]
    assert result["additional_guidance"] == "Include all main points."


def test_validator_with_empty_specific_guidance(validator):
    """Test validation when no specific guidance is provided."""
    task_description = "Summarize the text."
    generated_response = "Basic summary."

    from class_factory.utils.llm_validator import ValidatorInterimResponse

    # Create a real ValidatorInterimResponse object with numeric values
    expected_response = {
        "accuracy": 7.0,
        "completeness": 7.0,
        "consistency": 7.0,
        "reasoning": "Adequate response.",
        "additional_guidance": ""
    }

    # Mock chain that returns a real ValidatorInterimResponse
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = ValidatorInterimResponse(**expected_response)
    with patch('class_factory.utils.llm_validator.ChatPromptTemplate.from_messages') as mock_template:
        mock_template.return_value.__or__.return_value = mock_chain
        result = validator.validate(task_description, generated_response)

    assert isinstance(result, dict)
    assert "overall_score" in result
    assert "status" in result


def test_validator_prompt_template_format(validator):
    """Test that the prompt template contains all required components."""
    task_description = "Test task"
    generated_response = "Test response"

    # Patch the chain to return a valid response
    from class_factory.utils.llm_validator import ValidatorInterimResponse
    expected_response = {
        "accuracy": 8.0,
        "completeness": 8.0,
        "consistency": 8.0,
        "reasoning": "Good response.",
        "additional_guidance": ""
    }
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = ValidatorInterimResponse(**expected_response)
    with patch('langchain.prompts.ChatPromptTemplate.from_messages') as mock_prompt_template:
        mock_prompt_template.return_value.__or__.return_value = mock_chain
        validator.validate(task_description, generated_response)

    # Verify the template was called with a string containing required elements
    template_str = mock_prompt_template.call_args[0][0]
    template_text = template_str[1].prompt.template
    assert "accuracy" in template_text
    assert "completeness" in template_text
    assert "consistency" in template_text
    assert "reasoning" in template_text
    assert "additional_guidance" in template_text


def test_validator_retry_functionality(validator, caplog):
    """Test the retry decorator on validation method."""
    task_description = "Test task"
    generated_response = "Test response"

    # Mock chain to fail with JSON error first, then succeed
    from class_factory.utils.llm_validator import ValidatorInterimResponse
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [
        json.JSONDecodeError("Test error", "test", 0),  # First call fails
        ValidatorInterimResponse(
            accuracy=8.0,
            completeness=8.0,
            consistency=8.0,
            reasoning="Good response",
            additional_guidance=""
        )
    ]

    # Patch ChatPromptTemplate to ensure it produces the expected mock chain for both pipes
    with patch('langchain.prompts.ChatPromptTemplate.from_messages') as mock_prompt_template:
        mock_template = MagicMock()
        mock_template.__or__.return_value = mock_chain
        mock_prompt_template.return_value = mock_template
        result = validator.validate(task_description, generated_response)

    assert isinstance(result, dict)
    assert result["status"] == 1
    assert any('error' in record.message.lower() for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__])
