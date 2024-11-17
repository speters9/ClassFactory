import logging
import shutil
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from class_factory.quiz_maker.quiz_to_app import (load_data, next_question,
                                                  prev_question, quiz_app,
                                                  submit_answer)


@pytest.fixture
def mock_paths():
    # Create temporary directories
    temp_dirs = {
        "syllabus_path": Path(tempfile.mkdtemp()) / "syllabus.txt",
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
def mock_quiz_app():
    # Create mock components
    mock_radio = MagicMock()
    mock_textbox = MagicMock()
    mock_button = MagicMock()
    mock_state = MagicMock()
    mock_row = MagicMock()
    mock_markdown = MagicMock()

    # Create a mock Blocks instance with click method
    mock_blocks = MagicMock()
    mock_blocks.launch = MagicMock()
    mock_blocks.__enter__ = MagicMock(return_value=mock_blocks)
    mock_blocks.__exit__ = MagicMock(return_value=None)

    # Configure mock button click behavior
    mock_button.click.return_value = MagicMock()

    # Create mock classes that return our mock instances
    mock_blocks_class = MagicMock(return_value=mock_blocks)
    mock_radio_class = MagicMock(return_value=mock_radio)
    mock_textbox_class = MagicMock(return_value=mock_textbox)
    mock_button_class = MagicMock(return_value=mock_button)
    mock_state_class = MagicMock(return_value=mock_state)
    mock_row_class = MagicMock(return_value=mock_row)
    mock_markdown_class = MagicMock(return_value=mock_markdown)

    with patch('class_factory.quiz_maker.quiz_to_app.gr.Blocks', mock_blocks_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.Radio', mock_radio_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.Textbox', mock_textbox_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.Button', mock_button_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.State', mock_state_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.Row', mock_row_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.Markdown', mock_markdown_class), \
            patch('pathlib.Path.mkdir') as mock_mkdir, \
            patch('qrcode.make') as mock_qr_make, \
            patch('class_factory.quiz_maker.quiz_to_app.print') as mock_print:

        yield {
            'mock_blocks': mock_blocks,
            'mock_radio': mock_radio,
            'mock_button': mock_button,
            'mock_textbox': mock_textbox,
            'mock_state': mock_state,
            'mock_mkdir': mock_mkdir,
            'mock_qr_make': mock_qr_make,
            'mock_print': mock_print
        }


# Test the load_data function
def test_load_data(mock_paths):
    with patch('pandas.read_excel') as mock_read_excel:
        mock_df = pd.DataFrame({
            'question': ['What is Python?', 'What is Pandas?'],
            'A)': ['A programming language', 'A data analysis library'],
            'B)': ['A snake', 'A database'],
            'correct_answer': ['A)', 'A)']
        })
        mock_read_excel.return_value = mock_df

        quiz_data = load_data(mock_paths['output_dir'] / 'quiz.xlsx')
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

# Test next_question functionality


def test_next_question(mock_quiz_app, mock_paths):
    quiz_data = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'A)': ['A programming language', 'A data analysis library'],
        'B)': ['A snake', 'A database'],
        'correct_answer': ['A)', 'A)']
    })

    quiz_app(quiz_data, save_results=False, output_dir=mock_paths['output_dir'])
    mock_quiz_app['mock_blocks'].launch.assert_called_once()


def test_prev_question(mock_quiz_app, mock_paths):
    quiz_data = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'A)': ['A programming language', 'A data analysis library'],
        'B)': ['A snake', 'A database'],
        'correct_answer': ['A)', 'A)']
    })

    quiz_app(quiz_data, save_results=False, output_dir=mock_paths['output_dir'])
    mock_quiz_app['mock_blocks'].launch.assert_called_once()


def test_quiz_app_no_save(mock_paths):
    quiz_data = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'A)': ['A programming language', 'A data analysis library'],
        'B)': ['A snake', 'A database'],
        'correct_answer': ['A)', 'A)']
    })

    # Mock all Gradio components
    mock_radio = MagicMock()
    mock_textbox = MagicMock()
    mock_button = MagicMock()
    mock_state = MagicMock()
    mock_row = MagicMock()
    mock_markdown = MagicMock()

    # Create mock Blocks with context manager methods
    mock_blocks = MagicMock()
    mock_blocks.launch = MagicMock()
    mock_blocks.__enter__ = MagicMock(return_value=mock_blocks)
    mock_blocks.__exit__ = MagicMock(return_value=None)

    # Configure mock button click behavior
    mock_button.click.return_value = MagicMock()

    # Create mock classes
    mock_blocks_class = MagicMock(return_value=mock_blocks)
    mock_radio_class = MagicMock(return_value=mock_radio)
    mock_textbox_class = MagicMock(return_value=mock_textbox)
    mock_button_class = MagicMock(return_value=mock_button)
    mock_state_class = MagicMock(return_value=mock_state)
    mock_row_class = MagicMock(return_value=mock_row)
    mock_markdown_class = MagicMock(return_value=mock_markdown)

    with patch('class_factory.quiz_maker.quiz_to_app.gr.Blocks', mock_blocks_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.Radio', mock_radio_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.Textbox', mock_textbox_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.Button', mock_button_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.State', mock_state_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.Row', mock_row_class), \
            patch('class_factory.quiz_maker.quiz_to_app.gr.Markdown', mock_markdown_class), \
            patch('class_factory.quiz_maker.quiz_to_app.qrcode.make') as mock_qr_make, \
            patch('pathlib.Path.mkdir') as mock_mkdir:

        quiz_app(quiz_data, save_results=False, share=False, output_dir=mock_paths['output_dir'])
        mock_blocks.launch.assert_called_once()
        mock_qr_make.assert_not_called()


@pytest.mark.slow
def test_quiz_app_save(mock_quiz_app, mock_paths):
    quiz_data = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'A)': ['A programming language', 'A data analysis library'],
        'B)': ['A snake', 'A database'],
        'correct_answer': ['A)', 'A)']
    })

    quiz_app(quiz_data, save_results=True, output_dir=mock_paths['output_dir'])
    mock_quiz_app['mock_blocks'].launch.assert_called_once()
    mock_quiz_app['mock_mkdir'].assert_called()
    mock_quiz_app['mock_qr_make'].assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
