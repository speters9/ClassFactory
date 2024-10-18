import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import gradio as gr
import pandas as pd
import pytest

from src.quiz_maker.quiz_to_app import (load_data, next_question,
                                        prev_question, quiz_app, submit_answer)


# Test the load_data function
def test_load_data():
    with patch('pandas.read_excel') as mock_read_excel:
        mock_df = pd.DataFrame({
            'question': ['What is Python?', 'What is Pandas?'],
            'A)': ['A programming language', 'A data analysis library'],
            'B)': ['A snake', 'A database'],
            'correct_answer': ['A)', 'A)']
        })
        mock_read_excel.return_value = mock_df

        quiz_data = load_data(Path('fake/path/to/quiz.xlsx'))
        assert len(quiz_data) == 2
        assert 'question' in quiz_data.columns

# Test the submit_answer function


def test_submit_answer_correct():
    quiz_data = pd.DataFrame({
        'question': ['What is Python?'],
        'A)': ['A programming language'],
        'B)': ['A snake'],
        'correct_answer': ['A']
    })

    user_id = 'testuser'
    feedback = submit_answer(current_index=0, user_answer='A programming language', quiz_data=quiz_data,
                             user_id=user_id, save_results=False, output_dir=None)
    assert feedback == "Question 1: Correct!"


def test_submit_answer_incorrect():
    quiz_data = pd.DataFrame({
        'question': ['What is Python?'],
        'A)': ['A programming language'],
        'B)': ['A snake'],
        'correct_answer': ['A']
    })

    user_id = 'testuser'
    feedback = submit_answer(current_index=0, user_answer='A snake', quiz_data=quiz_data,
                             user_id=user_id, save_results=False, output_dir=None)
    assert feedback == "Question 1: Incorrect. The correct answer was: A programming language."


def test_next_question():
    quiz_data = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'A)': ['A programming language', 'A data analysis library'],
        'B)': ['A snake', 'A database'],
        'C)': [None, None],  # Ensure 'C)' and 'D)' are either present or set to None
        'D)': [None, None],
        'correct_answer': ['A)', 'A)']
    })

    question, feedback, index = next_question(current_index=0, quiz_data=quiz_data)
    assert index == 1  # Moves to the next question
    assert "What is Pandas?" in question['label']


def test_prev_question():
    quiz_data = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'A)': ['A programming language', 'A data analysis library'],
        'B)': ['A snake', 'A database'],
        'C)': [None, None],  # Ensure 'C)' and 'D)' are either present or set to None
        'D)': [None, None],
        'correct_answer': ['A)', 'A)']
    })

    question, feedback, index = prev_question(current_index=1, quiz_data=quiz_data)
    assert index == 0  # Moves back to the previous question
    assert "What is Python?" in question['label']


# Test the quiz_app function with save_results=False
def test_quiz_app_no_save():
    quiz_data = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'A)': ['A programming language', 'A data analysis library'],
        'B)': ['A snake', 'A database'],
        'correct_answer': ['A)', 'A)']
    })

    # Mock Gradio's Blocks and QR code creation
    with patch.object(gr.Blocks, 'launch', autospec=True) as mock_launch, \
            patch('qrcode.make') as mock_qr_make, \
            patch('pathlib.Path.mkdir') as mock_mkdir:

        # Run quiz_app with save_results=False
        quiz_app(quiz_data, save_results=False, output_dir='fake/output/dir')

        # Assert the launch method was called once
        mock_launch.assert_called_once()

        # Assert QR code generation was NOT called
        mock_qr_make.assert_not_called()


def test_quiz_app_no_url():
    quiz_data = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'A)': ['A programming language', 'A data analysis library'],
        'B)': ['A snake', 'A database'],
        'correct_answer': ['A)', 'A)']
    })

    # Adjust the import path to your module
    import quiz_maker.quiz_to_app  # Replace with the actual module path

    # Mock Path.mkdir and qrcode.make to prevent file operations
    with patch('pathlib.Path.mkdir') as mock_mkdir, \
            patch('qrcode.make') as mock_qr_make, \
            patch('quiz_maker.quiz_to_app.print') as mock_print:
        # Run quiz_app with save_results=True
        quiz_app(quiz_data, save_results=True, output_dir='fake/output/dir')

        # Ensure that qrcode.make was not called since no URL was generated
        mock_qr_make.assert_called_once()

        # Ensure that Path.mkdir was not called since no QR code was saved
        mock_mkdir.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
