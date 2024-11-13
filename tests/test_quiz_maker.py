import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
import torch
from sentence_transformers import SentenceTransformer

from class_factory.quiz_maker.quiz_viz import (generate_dashboard,
                                               generate_html_report)
from class_factory.quiz_maker.QuizMaker import QuizMaker
from class_factory.utils.tools import reset_loggers

reset_loggers()
logging.basicConfig(level=logging.WARNING)

# Base Fixtures


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.temperature.return_value = 0.6
    llm.invoke.return_value = {
        'multiple_choice': [{
            'question': 'Test question?',
            'A)': 'Option A',
            'B)': 'Option B',
            'correct_answer': 'A'
        }]
    }
    return llm


@pytest.fixture
def mock_paths(tmp_path):
    syllabus_path = tmp_path / "syllabus.docx"
    reading_dir = tmp_path / "readings"
    output_dir = tmp_path / "output"
    reading_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return syllabus_path, reading_dir, output_dir


@pytest.fixture
def base_quiz_maker(mock_llm, mock_paths):
    """Base QuizMaker fixture with common mocks."""
    syllabus_path, reading_dir, output_dir = mock_paths
    with patch.object(QuizMaker, '_validate_file_path', return_value=syllabus_path), \
            patch.object(QuizMaker, '_validate_dir_path', side_effect=lambda p, n: p):
        return QuizMaker(
            llm=mock_llm,
            syllabus_path=syllabus_path,
            reading_dir=reading_dir,
            output_dir=output_dir,
            prior_quiz_path=output_dir / "prior_quizzes",
            lesson_range=range(1, 5),
            course_name="Test Course",
            device='cpu'
        )

# Specialized Fixtures


@pytest.fixture
def quiz_maker_with_encoder(base_quiz_maker):
    """QuizMaker fixture with mocked encoder."""
    with patch.object(SentenceTransformer, 'encode') as mock_encode:
        def mock_encode_side_effect(texts, convert_to_tensor=True, device='cpu', **kwargs):
            return torch.randn(len(texts), 384).to(device)
        mock_encode.side_effect = mock_encode_side_effect
        yield base_quiz_maker


@pytest.fixture
def quiz_maker_with_chain(base_quiz_maker):
    """QuizMaker fixture with mocked quiz chain."""
    with patch.object(base_quiz_maker, '_build_quiz_chain') as mock_build_chain:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            'multiple_choice': [{
                'question': 'Test question?',
                'A)': 'Option A',
                'B)': 'Option B',
                'correct_answer': 'A',
                'type': 'multiple_choice'
            }]
        }
        mock_build_chain.return_value = mock_chain

        base_quiz_maker.validator = MagicMock()
        base_quiz_maker.validator.validate.return_value = {
            "evaluation_score": 8.0,
            "status": 1,
            "reasoning": "The question is accurate and relevant.",
            "additional_guidance": ""
        }

        # Ensure `lesson_range` is a single lesson
        base_quiz_maker.lesson_range = range(1, 2)
        base_quiz_maker.llm = mock_llm

        yield base_quiz_maker


@pytest.fixture
def quiz_maker_with_validator(base_quiz_maker):
    """QuizMaker fixture with mocked validator."""
    base_quiz_maker.validator = MagicMock()
    base_quiz_maker.validator.validate.return_value = {
        "evaluation_score": 8.0,
        "status": 1,
        "reasoning": "The question is accurate and relevant.",
        "additional_guidance": ""
    }
    return base_quiz_maker

# Test Data Fixtures


@pytest.fixture
def sample_quiz_data():
    """Fixture for sample quiz data."""
    return [
        {
            'question': 'What is Python?',
            'A)': 'A programming language',
            'B)': 'A type of snake',
            'C)': '',
            'D)': '',
            'correct_answer': 'A programming language'
        },
        {
            'question': 'What is the capital of France?',
            'A)': 'Berlin',
            'B)': 'Madrid',
            'C)': 'Paris',
            'D)': 'London',
            'correct_answer': 'Paris'
        }
    ]


@pytest.fixture
def sample_quiz_results():
    """Fixture for quiz results data."""
    return pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3', 'user1'],
        'question': ['What is Python?', 'What is Python?', 'What is Pandas?', 'What is Python?'],
        'user_answer': ['A programming language', 'A snake', 'A data analysis library', 'A programming language'],
        'correct_answer': ['A programming language', 'A programming language', 'A data analysis library', 'A programming language'],
        'is_correct': [True, False, True, True],
        'timestamp': [
            '2024-10-21T16:07:43.232332',
            '2024-10-21T16:08:00.232332',
            '2024-10-21T16:09:00.232332',
            '2024-10-21T16:09:10.232332'
        ]
    })


# Quiz Generation Tests

def test_make_a_quiz(quiz_maker_with_chain):
    with patch('class_factory.utils.load_documents.load_docx_syllabus') as mock_load_syllabus, \
            patch.object(QuizMaker, '_check_question_similarity', return_value=[]):

        mock_load_syllabus.return_value = ["Objective 1", "Objective 2"]
        result = quiz_maker_with_chain.make_a_quiz()

        assert len(result) == 1
        assert result[0]['question'] == 'Test question?'
        assert result[0]['correct_answer'] == 'A'
        assert result[0]['type'] == 'multiple_choice'


@pytest.mark.slow
def test_json_decode_error_retry(quiz_maker_with_chain):
    # Mock the `chain.invoke` method directly
    with patch.object(quiz_maker_with_chain, '_build_quiz_chain') as mock_build_chain:
        mock_chain = MagicMock()
        # Simulate the side effects of retries on the chain invoke
        mock_chain.invoke.side_effect = [
            'invalid json',  # First call returns invalid JSON
            {'incomplete': {'question': 'Wrong format?'}},  # Second call, incorrect structure
            {  # Third call, valid response
                'multiple_choice': [{
                    'question': 'Test question?',
                    'A)': 'Option A',
                    'B)': 'Option B',
                    'correct_answer': 'A'
                }]
            }
        ]
        mock_build_chain.return_value = mock_chain

        # Mock `load_docx_syllabus` to avoid the actual file dependency
        with patch('class_factory.utils.load_documents.load_docx_syllabus', return_value=["Objective"]):
            result = quiz_maker_with_chain.make_a_quiz()

            # Validate that `chain.invoke` was called three times as expected
            assert mock_chain.invoke.call_count == 3, f"Expected 3 calls, got {mock_chain.invoke.call_count}"

            # Validate the returned result
            assert len(result) == 1
            assert result[0]['question'] == 'Test question?'


# Validation Tests
def test_validate_questions(base_quiz_maker, sample_quiz_data):
    fixed_quiz_data = base_quiz_maker._validate_questions(sample_quiz_data)
    assert fixed_quiz_data[0]['correct_answer'] == 'A'
    assert fixed_quiz_data[1]['correct_answer'] == 'C'


def test_validate_llm_response(quiz_maker_with_validator):
    quiz_questions = {
        "multiple_choice": [{
            "question": "What is 2 + 2?",
            "A)": "3",
            "B)": "4",
            "correct_answer": "B"
        }]
    }

    val_response = quiz_maker_with_validator._validate_llm_response(
        quiz_questions=quiz_questions,
        objectives="Understand basic arithmetic",
        readings="Basic math content",
        prior_quiz_questions=["What is 3 + 3?"],
        difficulty_level=5,
        additional_guidance="Focus on comprehension"
    )

    quiz_maker_with_validator.validator.validate.assert_called_once()
    assert val_response["evaluation_score"] == 8.0

# Similarity Check Tests


def test_check_question_similarity(quiz_maker_with_encoder):
    generated_questions = ['What is the capital of France?']
    quiz_maker_with_encoder.prior_quiz_questions = ['What is the capital of Germany?']
    flagged_questions = quiz_maker_with_encoder._check_question_similarity(generated_questions, threshold=0.5)
    assert isinstance(flagged_questions, list)

# Save/Export Tests


@patch('pandas.DataFrame.to_excel')
def test_save_quiz(mock_to_excel, base_quiz_maker):
    mock_quiz = [{
        'type': 'multiple_choice',
        'question': 'What is 2 + 2?',
        'A)': '3',
        'B)': '4',
        'C)': '5',
        'D)': '6',
        'correct_answer': 'B'
    }]

    base_quiz_maker.save_quiz(mock_quiz)
    mock_to_excel.assert_called_once()
    args, kwargs = mock_to_excel.call_args
    assert 'l1_4_quiz.xlsx' in str(args[0])


@patch('class_factory.quiz_maker.QuizMaker.Presentation')
def test_save_quiz_to_ppt(mock_presentation_class, base_quiz_maker):
    mock_prs = MagicMock()
    mock_presentation_class.return_value = mock_prs

    mock_quiz = [{
        'type': 'multiple_choice',
        'question': 'What is 2 + 2?',
        'A)': '3',
        'B)': '4',
        'correct_answer': 'B'
    }]

    base_quiz_maker.save_quiz_to_ppt(mock_quiz)
    mock_prs.save.assert_called_once()

# Assessment Tests


@pytest.mark.slow
def test_assess_quiz_results(base_quiz_maker, sample_quiz_results):
    # Patch the exact path where these functions are used in assess_quiz_results
    with patch('class_factory.quiz_maker.QuizMaker.generate_html_report') as mock_html_report, \
            patch('class_factory.quiz_maker.QuizMaker.generate_dashboard') as mock_dashboard:

        # Set up mock responses to avoid running actual Dash server or report generation
        mock_html_report.return_value = None
        mock_dashboard.return_value = None

        # Run the assessment function
        summary = base_quiz_maker.assess_quiz_results(quiz_data=sample_quiz_results)

        # Assertions to check if mocks were called
        mock_html_report.assert_called_once()
        mock_dashboard.assert_called_once()

        # Validate the summary DataFrame
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0

# Visualization Tests


@pytest.mark.slow
@patch('class_factory.quiz_maker.quiz_viz.Dash.run_server')
@patch('class_factory.quiz_maker.quiz_viz.px.bar')
@patch('class_factory.quiz_maker.quiz_viz.create_question_figure')
def test_generate_dashboard(mock_create_figure, mock_px_bar, mock_run_server, sample_quiz_results):
    summary = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'Total Responses': [3, 2],
        'Correct Responses': [2, 2],
        'Percent Correct': [66.7, 100]
    })

    mock_figure = MagicMock()
    mock_create_figure.return_value = mock_figure

    generate_dashboard(sample_quiz_results, summary, test_mode=True)

    mock_create_figure.assert_called()
    assert mock_create_figure.call_count == 2
    mock_run_server.assert_not_called()


def test_generate_html_report(sample_quiz_results, tmp_path):
    mock_env = MagicMock()
    mock_template = MagicMock()
    mock_env.from_string.return_value = mock_template

    summary = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'Total Responses': [3, 2],
        'Correct Responses': [2, 2],
        'Percent Correct': [66.7, 100]
    })

    mock_figure = MagicMock()
    mock_figure.to_html.return_value = "<div>Mock Plot</div>"

    with patch('jinja2.Environment', return_value=mock_env), \
            patch('builtins.open', mock_open()), \
            patch('class_factory.quiz_maker.quiz_viz.create_question_figure', return_value=mock_figure):

        generate_html_report(sample_quiz_results, summary, tmp_path)

        mock_env.from_string.assert_called_once()
        mock_template.render.assert_called_once()
        assert mock_figure.to_html.call_count == len(sample_quiz_results['question'].unique())


if __name__ == "__main__":
    pytest.main([__file__])
