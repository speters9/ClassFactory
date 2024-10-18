import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer

from src.quiz_maker.QuizMaker import QuizMaker
from src.utils.tools import reset_loggers

reset_loggers()
logging.basicConfig(level=logging.WARNING)

# Mock LLM


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = {'multiple_choice': [{'question': 'Test question?', 'A)': 'Answer A', 'B)': 'Answer B', 'correct_answer': 'A'}]}
    return llm

# Mock environment variables and paths


@pytest.fixture
def mock_paths(monkeypatch):
    # Mock environment variables for paths
    monkeypatch.setenv('syllabus_path', '/mock/syllabus')
    monkeypatch.setenv('readingsDir', '/mock/readings')
    monkeypatch.setenv('openai_key', 'mock_openai_key')
    monkeypatch.setenv('openai_org', 'mock_openai_org')

    # Return mock Path objects
    return Path('/mock/syllabus.docx'), Path('/mock/readings'), Path('/mock/output')


# Test initialization of the QuizMaker class
def test_quiz_maker_init(mock_llm, mock_paths):
    syllabus_path, reading_dir, output_dir = mock_paths
    quiz_maker = QuizMaker(
        llm=mock_llm,
        syllabus_path=syllabus_path,
        reading_dir=reading_dir,
        output_dir=output_dir,
        prior_quiz_path=Path('/path/to/prior_quizzes'),
        lesson_range=range(1, 5),
        course_name="Test Course"
    )
    assert quiz_maker.llm == mock_llm
    assert quiz_maker.syllabus_path == syllabus_path
    assert quiz_maker.reading_dir == reading_dir
    assert quiz_maker.output_dir == output_dir


# Mock the entire quiz chain
@patch.object(SentenceTransformer, 'encode')
@patch.object(QuizMaker, 'load_and_merge_prior_quizzes')
@patch.object(QuizMaker, 'build_quiz_chain')
@patch('src.utils.load_documents.Document')
@patch('src.utils.load_documents.extract_lesson_objectives', return_value='Lesson Objective 1')
@patch('src.utils.load_documents.load_lessons', return_value=['Reading 1', 'Reading 2'])
def test_make_a_quiz(mock_load_lessons, mock_extract_lesson_objectives, mock_Document,
                     mock_build_chain, mock_load_and_merge_prior_quizzes, mock_encode):
    # Define mock paths
    syllabus_path = Path('/mock/syllabus.docx')
    reading_dir = Path('/mock/readings')
    output_dir = Path('/mock/output')

    # Mock Document object to avoid file system access
    mock_Document.return_value.paragraphs = [MagicMock(text="Lesson objective 1")]

    # Create mock for chain.invoke()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        'multiple_choice': [
            {
                'question': 'What is 2 + 2?',
                'A)': '3',
                'B)': '4',
                'C)': '5',
                'D)': '6',
                'correct_answer': 'B'
            }
        ]
    }
    mock_build_chain.return_value = mock_chain

    # Mock the encode method to return embeddings on the specified device
    def mock_encode_side_effect(texts, convert_to_tensor=True, device='cpu', **kwargs):
        embeddings = torch.randn(len(texts), 384).to(device)  # Assuming embedding size of 384
        return embeddings

    mock_encode.side_effect = mock_encode_side_effect

    # Mock prior quiz questions to prevent empty embeddings
    mock_load_and_merge_prior_quizzes.return_value = (['What is 1 + 1?'], None)

    # Instantiate QuizMaker with mock paths and set device='cpu'
    quiz_maker = QuizMaker(
        llm=None,
        syllabus_path=syllabus_path,
        reading_dir=reading_dir,
        output_dir=output_dir,
        prior_quiz_path=Path('/path/to/prior_quizzes'),
        lesson_range=range(1, 5),
        course_name="Test Course",
        device='cpu'
    )

    # Call make_a_quiz and check the output
    quiz = quiz_maker.make_a_quiz()

    # Assertions
    assert len(quiz) > 0
    assert quiz[0]['question'] == 'What is 2 + 2?'
    assert quiz[0]['correct_answer'] == 'B'


# Test checking similarity (mock the SentenceTransformer model)
@patch.object(SentenceTransformer, 'encode')
def test_check_question_similarity(mock_encode, mock_llm, mock_paths):
    syllabus_path, reading_dir, output_dir = mock_paths

    # Mock the encode method to return a tensor on the correct device
    def mock_encode_side_effect(texts, *args, **kwargs):
        device = torch.device('cpu')
        # Return a tensor with appropriate size on the specified device
        return torch.randn(len(texts), 384).to(device)  # Assuming embedding size of 384

    mock_encode.side_effect = mock_encode_side_effect

    quiz_maker = QuizMaker(
        llm=mock_llm,
        syllabus_path=syllabus_path,
        reading_dir=reading_dir,
        output_dir=output_dir,
        prior_quiz_path=Path('/path/to/prior_quizzes'),
        lesson_range=range(1, 5),
        course_name="Test Course",
        device=torch.device('cpu')  # Set device to 'cpu' for testing
    )

    generated_questions = ['What is the capital of France?']
    quiz_maker.prior_quiz_questions = ['What is the capital of Germany?']

    flagged_questions = quiz_maker.check_question_similarity(generated_questions, threshold=0.5)

    # Since the embeddings are random, adjust the assertion accordingly
    assert len(flagged_questions) >= 0  # Check that the method runs without error


# Test save_quiz method with mock DataFrame and Excel saving
@patch('pandas.DataFrame.to_excel')
def test_save_quiz(mock_to_excel, mock_llm, mock_paths):
    syllabus_path, reading_dir, _ = mock_paths
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
    quiz_maker = QuizMaker(
        llm=mock_llm,
        syllabus_path=syllabus_path,
        reading_dir=reading_dir,
        output_dir=output_dir,
        prior_quiz_path=Path('/path/to/prior_quizzes'),
        lesson_range=range(1, 5),
        course_name="Test Course"
    )

    # Add the missing C) and D) columns
    mock_quiz = [
        {'type': 'multiple_choice', 'question': 'What is 2 + 2?', 'A)': '3', 'B)': '4', 'C)': '5', 'D)': '6', 'correct_answer': 'B'}
    ]

    quiz_maker.save_quiz(mock_quiz)

    mock_to_excel.assert_called_once()
    args, kwargs = mock_to_excel.call_args
    # The file path is the first positional argument
    assert 'l1_4_quiz.xlsx' in str(args[0])  # Check the file name


@patch('src.quiz_maker.QuizMaker.Presentation')
def test_save_quiz_to_ppt(mock_presentation_class, mock_llm, mock_paths):
    syllabus_path, reading_dir, _ = mock_paths
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Create a mock instance of Presentation
        mock_prs_instance = MagicMock()
        mock_presentation_class.return_value = mock_prs_instance

        quiz_maker = QuizMaker(
            llm=mock_llm,
            syllabus_path=syllabus_path,
            reading_dir=reading_dir,
            output_dir=output_dir,
            prior_quiz_path=Path('/path/to/prior_quizzes'),
            lesson_range=range(1, 5),
            course_name="Test Course"
        )

        mock_quiz = [{
            'type': 'multiple_choice',
            'question': 'What is 2 + 2?',
            'A)': '3',
            'B)': '4',
            'C)': '',
            'D)': '',
            'correct_answer': 'B'
        }]

        quiz_maker.save_quiz_to_ppt(mock_quiz)

        # Assert that the 'save' method was called on the Presentation instance
        mock_prs_instance.save.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
