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

    feedback = submit_answer(current_index=0, user_answer='A programming language', quiz_data=quiz_data)
    assert feedback == "Question 1: Correct!"


def test_submit_answer_incorrect():
    quiz_data = pd.DataFrame({
        'question': ['What is Python?'],
        'A)': ['A programming language'],
        'B)': ['A snake'],
        'correct_answer': ['A']
    })

    feedback = submit_answer(current_index=0, user_answer='A snake', quiz_data=quiz_data)
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


def test_quiz_app():
    quiz_data = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'A)': ['A programming language', 'A data analysis library'],
        'B)': ['A snake', 'A database'],
        'correct_answer': ['A)', 'A)']
    })

    # Mock Gradio's interface components
    # Patch the gradio interface's launch method
    with patch.object(gr.Blocks, 'launch', autospec=True) as mock_launch:
        quiz_app(quiz_data)

        # Check if the launch method was called
        mock_launch.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
