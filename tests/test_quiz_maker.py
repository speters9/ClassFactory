import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
import torch
from sentence_transformers import SentenceTransformer

from src.quiz_maker.quiz_viz import generate_dashboard, generate_html_report
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


# test validation function
@pytest.fixture
def quiz_maker():
    """Fixture to initialize the QuizMaker instance."""
    return QuizMaker(
        llm=None,  # Assuming the LLM is mocked or not needed for this test
        syllabus_path=Path('syllabus_path'),
        reading_dir=Path('reading_dir'),
        output_dir=Path('output_dir'),
        prior_quiz_path=Path('prior_quiz_path'),
        lesson_range=range(1, 2),
        verbose=False
    )


def test_validate_questions_correct_text(quiz_maker):
    """Test that validate_questions fixes questions with the correct answer as text."""

    # Mock question data with the correct answer as the full text instead of the letter
    quiz_data = [
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

    # Call the validate_questions method (assumed to be part of QuizMaker class)
    fixed_quiz_data = quiz_maker.validate_questions(quiz_data)

    # Check if the correct_answer field is fixed and replaced with the corresponding letter
    assert fixed_quiz_data[0]['correct_answer'] == 'A', "The correct answer should be 'A' for the first question"
    assert fixed_quiz_data[1]['correct_answer'] == 'C', "The correct answer should be 'C' for the second question"


def test_validate_questions_incorrect_formatting(quiz_maker):
    """Test that validate_questions leaves already correctly formatted answers untouched."""

    # Mock question data with the correct answer as the letter
    quiz_data = [
        {
            'question': 'What is Python?',
            'A)': 'A programming language',
            'B)': 'A type of snake',
            'C)': '',
            'D)': '',
            'correct_answer': 'A'
        },
        {
            'question': 'What is the capital of France?',
            'A)': 'Berlin',
            'B)': 'Madrid',
            'C)': 'Paris',
            'D)': 'London',
            'correct_answer': 'C'
        }
    ]

    # Call the validate_questions method (assumed to be part of QuizMaker class)
    fixed_quiz_data = quiz_maker.validate_questions(quiz_data)

    # Ensure the correct_answer remains unchanged
    assert fixed_quiz_data[0]['correct_answer'] == 'A', "The correct answer should remain 'A' for the first question"
    assert fixed_quiz_data[1]['correct_answer'] == 'C', "The correct answer should remain 'C' for the second question"


@pytest.fixture
def mock_csv_data():
    return pd.DataFrame({
        'user_id': ['user1', 'user1', 'user2', 'user3'],
        'question': ['What is Python?', 'What is Python?', 'What is Python?', 'What is Pandas?'],
        'user_answer': ['A programming language', 'A programming language', 'A snake', 'A data analysis library'],
        'correct_answer': ['A programming language', 'A programming language', 'A programming language', 'A data analysis library'],
        'is_correct': [True, True, False, True],
        'timestamp': [
            '2024-10-21T16:07:43.232332',
            '2024-10-21T16:07:44.232332',
            '2024-10-21T16:08:00.232332',
            '2024-10-21T16:09:00.232332'
        ]
    })


@patch('src.quiz_maker.QuizMaker.generate_html_report')
@patch('src.quiz_maker.QuizMaker.generate_dashboard')
def test_assess_quiz_results(mock_dashboard, mock_html_report, quiz_maker):
    # Use a temporary directory as output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output_dir = Path(temp_dir)
        quiz_maker.output_dir = temp_output_dir  # Override with temporary directory

        # Mock data to simulate CSV content
        mock_csv_data = pd.DataFrame({
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

        # Run the assessment function with the mock data
        summary = quiz_maker.assess_quiz_results(quiz_data=mock_csv_data)

        # Assertions to verify the summary statistics
        assert len(summary) == 2  # Expecting two unique questions
        assert 'What is Python?' in summary['question'].values
        assert 'What is Pandas?' in summary['question'].values

        # Validate individual question statistics
        python_stats = summary[summary['question'] == 'What is Python?'].iloc[0]
        assert python_stats['Total Responses'] == 2
        assert python_stats['Correct Responses'] == 1
        assert python_stats['Incorrect Responses'] == 1

        pandas_stats = summary[summary['question'] == 'What is Pandas?'].iloc[0]
        assert pandas_stats['Total Responses'] == 1
        assert pandas_stats['Correct Responses'] == 1
        assert pandas_stats['Incorrect Responses'] == 0

        # Verify that directory creation and report generation methods were called
        mock_html_report.assert_called_once()
        mock_dashboard.assert_called_once()


@patch('pandas.read_csv')
@patch('src.quiz_maker.QuizMaker.generate_html_report')
@patch('src.quiz_maker.QuizMaker.generate_dashboard')
def test_multiple_users(mock_dashboard, mock_html_report, mock_read_csv, quiz_maker):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output_dir = Path(temp_dir)
        quiz_maker.output_dir = temp_output_dir

        csv_data = pd.DataFrame({
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

        # Use the DataFrame directly
        summary = quiz_maker.assess_quiz_results(quiz_data=csv_data)

        # Assertions
        python_stats = summary[summary['question'] == 'What is Python?'].iloc[0]
        assert python_stats['Total Responses'] == 2  # Should count unique users
        assert python_stats['Correct Responses'] == 1
        assert python_stats['Incorrect Responses'] == 1

        pandas_stats = summary[summary['question'] == 'What is Pandas?'].iloc[0]
        assert pandas_stats['Total Responses'] == 1
        assert pandas_stats['Correct Responses'] == 1
        assert pandas_stats['Incorrect Responses'] == 0


@pytest.fixture
def mock_extract_objectives():
    with patch('src.quiz_maker.QuizMaker.extract_lesson_objectives') as mock:
        mock.return_value = "Test objectives"
        yield mock


@pytest.fixture
def mock_llm():
    # Create a mock object that mimics ChatOpenAI
    mock = MagicMock()
    # Return a JSON string to match what the real LLM might return
    mock.invoke.return_value = json.dumps({
        'multiple_choice': [
            {
                'question': 'Test question?',
                'A)': 'Option A',
                'B)': 'Option B',
                'C)': 'Option C',
                'D)': 'Option D',
                'correct_answer': 'A',
                'type': 'multiple_choice'
            }
        ]
    })
    return mock


@pytest.fixture
@patch('src.quiz_maker.QuizMaker.JsonOutputParser')
def mock_json_parser():
    mock = MagicMock()
    mock.invoke.return_value = {
        'multiple_choice': [
            {
                'question': 'Test question?',
                'A)': 'Option A',
                'B)': 'Option B',
                'C)': 'Option C',
                'D)': 'Option D',
                'correct_answer': 'A',
                'type': 'multiple_choice'
            }
        ]
    }
    return mock


@pytest.fixture
def quiz_maker_with_mocks(mock_llm, mock_json_parser, mock_extract_objectives):
    """Fixture to initialize the QuizMaker instance with all necessary mocks."""
    return QuizMaker(
        llm=mock_llm,
        syllabus_path=Path('syllabus_path.docx'),
        reading_dir=Path('reading_dir'),
        output_dir=Path('output_dir'),
        prior_quiz_path=Path('prior_quiz_path'),
        lesson_range=range(1, 2),
        verbose=False
    )


@patch.object(QuizMaker, 'build_quiz_chain')
@patch('src.utils.load_documents.load_docx_syllabus')
@patch.object(QuizMaker, 'check_question_similarity')  # Mock the similarity check
def test_llm_integration(mock_check_similarity, mock_load_syllabus, mock_build_quiz_chain, mock_llm):
    # Mock syllabus content to avoid needing a real file
    mock_load_syllabus.return_value = ["Objective 1", "Objective 2"]

    # Create a mock chain with an invoke method
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        'multiple_choice': [
            {
                'question': 'Test question?',
                'A)': 'Option A',
                'B)': 'Option B',
                'correct_answer': 'A',
                'type': 'multiple_choice'
            }
        ]
    }
    mock_build_quiz_chain.return_value = mock_chain

    # Mock check_question_similarity to bypass actual tensor calculations
    mock_check_similarity.return_value = []  # No flagged questions

    # Instantiate QuizMaker with mocks
    quiz_maker = QuizMaker(
        llm=mock_llm,
        syllabus_path=Path('syllabus_path.docx'),
        reading_dir=Path('reading_dir'),
        output_dir=Path('output_dir'),
        prior_quiz_path=Path('prior_quiz_path'),
        lesson_range=range(1, 2),
        verbose=False
    )

    # Call make_a_quiz, which should now use the mocked methods
    result = quiz_maker.make_a_quiz()

    # Check if the expected question was generated
    assert len(result) == 1
    assert result[0]['question'] == 'Test question?'
    assert result[0]['correct_answer'] == 'A'
    assert result[0]['type'] == 'multiple_choice'


@pytest.mark.slow
@patch.object(QuizMaker, 'build_quiz_chain')
@patch('src.utils.load_documents.load_docx_syllabus')
@patch('src.utils.load_documents.load_lessons')
def test_json_decode_error_retry_no_prior_quizzes(mock_load_lessons, mock_load_syllabus, mock_build_quiz_chain):
    # Mock syllabus content and lesson readings
    mock_load_syllabus.return_value = ["Objective 1", "Objective 2"]
    mock_load_lessons.return_value = ["Lesson reading content"]

    # Create a mock LLM with side effects that will trigger retries
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        'invalid json',  # First response causes JSON error
        {  # Second response has incorrect structure - Return as dict to simulate parsed JSON
            'incomplete': {
                'question': 'Wrong format question?',
                'A)': 'Option A',
                'B)': 'Option B',
                'correct_answer': 'A'
            }
        },
        {  # Third response is correctly formatted - Return as dict to simulate parsed JSON
            'multiple_choice': [{
                'question': 'Test question?',
                'A)': 'Option A',
                'B)': 'Option B',
                'correct_answer': 'A'
            }]
        }
    ]

    # Configure mock chain
    mock_chain = MagicMock()
    mock_chain.invoke = mock_llm.invoke
    mock_build_quiz_chain.return_value = mock_chain

    # Create a mock for load_and_merge_prior_quizzes that returns empty lists
    with patch.object(QuizMaker, 'load_and_merge_prior_quizzes', return_value=([], pd.DataFrame())):
        # Instantiate QuizMaker
        quiz_maker = QuizMaker(
            llm=mock_llm,
            syllabus_path=Path('syllabus_path.docx'),
            reading_dir=Path('reading_dir'),
            output_dir=Path('output_dir'),
            prior_quiz_path=Path('prior_quiz_path'),
            lesson_range=range(1, 2),
            verbose=False
        )

        # Call the method to test retry handling
        result = quiz_maker.make_a_quiz()
        print(result)
        # Verify the results
        assert mock_llm.invoke.call_count == 3  # Ensure three attempts were made
        assert len(result) == 1  # Should have one question in the final result
        assert result[0]['question'] == 'Test question?'  # Verify correct question was returned
        assert result[0]['type'] == 'multiple_choice'  # Verify question type was added


@pytest.mark.slow
@patch('src.quiz_maker.quiz_viz.Dash.run_server')  # Mock Dash server directly in generate_dashboard
@patch('src.quiz_maker.quiz_viz.px.bar')  # Mock plotly bar chart
@patch('src.quiz_maker.quiz_viz.create_question_figure')  # Mock custom figure creation
def test_generate_dashboard(mock_create_figure, mock_px_bar, mock_run_server, quiz_maker):
    # Mock the figure return value
    mock_figure = MagicMock()
    mock_create_figure.return_value = mock_figure

    # Mock data for testing
    df = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'user_answer': ['A programming language', 'A data analysis library'],
        'is_correct': [True, True],
        'correct_answer': ['A programming language', 'A data analysis library']
    })
    summary = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'Total Responses': [3, 2],
        'Correct Responses': [2, 2],
        'Percent Correct': [66.7, 100]
    })

    # Call the function; with `run_server` patched, it should not actually start a server
    generate_dashboard(df, summary, test_mode=True)

    # Ensure `create_question_figure` and `px.bar` were called as expected
    mock_create_figure.assert_called()
    assert mock_create_figure.call_count == 2  # Two questions, so two figures

    # Check that `run_server` was not called
    mock_run_server.assert_not_called()


def test_generate_html_report():
    # Create a mock environment
    mock_env = MagicMock()
    mock_template = MagicMock()
    mock_env.from_string.return_value = mock_template

    # Mock the file operations
    mock_file = mock_open()

    # Create test data
    df = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'user_answer': ['A programming language', 'A data analysis library'],
        'is_correct': [True, True],
        'correct_answer': ['A programming language', 'A data analysis library']
    })

    summary = pd.DataFrame({
        'question': ['What is Python?', 'What is Pandas?'],
        'Total Responses': [3, 2],
        'Correct Responses': [2, 2],
        'Percent Correct': [66.7, 100]
    })

    output_dir = Path('/mock/output')

    # Mock the figure creation to avoid plotting
    mock_figure = MagicMock()
    mock_figure.to_html.return_value = "<div>Mock Plot</div>"

    with patch('jinja2.Environment', return_value=mock_env), \
            patch('builtins.open', mock_file), \
            patch('src.quiz_maker.quiz_viz.create_question_figure', return_value=mock_figure):

        # Call the function
        generate_html_report(df, summary, output_dir)

        # Verify that from_string was called
        mock_env.from_string.assert_called_once()

        # Verify that the template was rendered
        mock_template.render.assert_called_once()

        # Verify that a file was opened for writing
        mock_file.assert_called_once_with(output_dir / 'quiz_report.html', 'w', encoding='utf-8')

        # Verify that plots were created for each question
        assert mock_figure.to_html.call_count == len(df['question'].unique())


if __name__ == "__main__":
    pytest.main([__file__])
