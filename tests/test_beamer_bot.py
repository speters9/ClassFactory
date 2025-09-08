import logging
import shutil
import tempfile
from itertools import chain, repeat
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from class_factory.beamer_bot.BeamerBot import BeamerBot  # Updated import path
from class_factory.utils.load_documents import \
    LessonLoader  # Updated import path


@pytest.fixture
def mock_llm():
    return Mock()


@pytest.fixture
def mock_paths():
    # Create temporary directories
    temp_dirs = {
        "syllabus_path": Path(tempfile.mkdtemp()) / "syllabus.txt",
        "reading_dir": Path(tempfile.mkdtemp()),
        "slide_dir": Path(tempfile.mkdtemp()),
        "output_dir": Path(tempfile.mkdtemp()),
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
def mock_lesson_loader(mock_paths):
    # Create mock readings within the reading directory
    lesson_dir = mock_paths["reading_dir"] / "L1"
    lesson_dir.mkdir(parents=True)
    (lesson_dir / "reading1.txt").write_text("Reading 1 content")
    (lesson_dir / "reading2.txt").write_text("Reading 2 content")

    return LessonLoader(
        syllabus_path=mock_paths["syllabus_path"],
        reading_dir=mock_paths["reading_dir"],
        slide_dir=mock_paths["slide_dir"]
    )


@pytest.fixture
def beamer_bot(mock_llm, mock_paths, mock_lesson_loader):
    with patch.object(BeamerBot, '_load_prior_lesson', return_value="Mocked previous lesson content"), \
            patch('pathlib.Path.is_file', return_value=True):
        return BeamerBot(
            lesson_no=1,
            lesson_loader=mock_lesson_loader,
            llm=mock_llm,
            output_dir=mock_paths["output_dir"],
            course_name="American Government"
        )


def test_beamer_bot_initialization(beamer_bot, mock_paths):
    assert beamer_bot.lesson_no == 1
    assert beamer_bot.lesson_loader.syllabus_path == mock_paths["syllabus_path"]
    assert beamer_bot.lesson_loader.reading_dir == mock_paths["reading_dir"]
    assert beamer_bot.lesson_loader.slide_dir == mock_paths["slide_dir"]
    assert beamer_bot.output_dir == mock_paths["output_dir"]
    assert isinstance(beamer_bot.lesson_loader, LessonLoader)
    assert beamer_bot.course_name == "American Government"


@patch('class_factory.utils.load_documents.LessonLoader.load_lessons', return_value={"1": ["Reading 1", "Reading 2"]})
def test_load_readings(mock_load_lessons, beamer_bot):
    """Test loading readings using the fixture."""
    # Call the `_load_readings` method
    readings_dict = beamer_bot._load_readings(beamer_bot.lesson_no)

    # Verify that `load_lessons` was called on the `LessonLoader` with the correct arguments
    mock_load_lessons.assert_called_once_with(lesson_number_or_range=range(beamer_bot.lesson_no, beamer_bot.lesson_no + 1))

    # Check that the returned dictionary has the expected structure
    expected_readings_dict = {"1": ["Reading 1", "Reading 2"]}
    assert readings_dict == expected_readings_dict


def test_format_readings_for_prompt(beamer_bot):
    """Test the `_format_readings_for_prompt` method for correct output formatting."""
    # Mock `_load_readings` to return a specific dictionary
    beamer_bot._load_readings = Mock(
        return_value={"1": ["Reading1", "Reading2"], "2": ["Reading3"]})

    # Call the `_format_readings_for_prompt` method
    formatted_readings = beamer_bot._format_readings_for_prompt()
    # Check that the formatted string matches the expected output
    expected_formatted = "Lesson 1, Reading 1:\nReading1\n\nLesson 1, Reading 2:\nReading2\n\nLesson 2, Reading 1:\nReading3\n"
    assert formatted_readings == expected_formatted


@patch('class_factory.utils.load_documents.LessonLoader.extract_lesson_objectives')
@patch('class_factory.utils.load_documents.LessonLoader.load_beamer_presentation')
@patch('class_factory.beamer_bot.BeamerBot.validate_latex')
def test_generate_slides(
    mock_validate_latex,
    mock_load_prior_lesson,
    mock_extract_objectives,
    beamer_bot
):
    """Test slide generation using the fixture."""
    # Set up return values
    mock_extract_objectives.return_value = "Test objectives"
    mock_load_prior_lesson.return_value = "Previous lesson content"
    mock_validate_latex.return_value = True

    # Create a mock BeamerSlides object
    from class_factory.beamer_bot.beamer_slides import BeamerSlides, Slide
    mock_slides = BeamerSlides(
        slides=[
            Slide(title="Test Slide", content="Test content", slide_type="objectives"),
            Slide(title="Summary", content="Summary content", slide_type="summary")
        ],
        title="Lesson 1",
        author="Test Author",
        institute="Test Institute",
        date="2024-01-01"
    )

    # Patch to_latex to return a known LaTeX string using patch.object
    mock_slides_latex = "\\title{Lesson 1}\n\\author{Test Author}\n\\institute{Test Institute}\n\\date{2024-01-01}\n\\begin{document}\n...slides...\\end{document}"

    # Use a MagicMock for the LLM output with a to_latex method
    mock_slides_latex = "\\title{Lesson 1}\n\\author{Test Author}\n\\institute{Test Institute}\n\\date{2024-01-01}\n\\begin{document}\n...slides...\\end{document}"
    mock_slides = MagicMock()
    mock_slides.to_latex.return_value = mock_slides_latex
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_slides

    with patch.object(BeamerBot, '_validate_llm_response') as mock_validator:
        mock_validator.return_value = {
            "evaluation_score": 8.5,
            "status": 1,
            "reasoning": "Slides are accurate and well-structured.",
            "additional_guidance": ""
        }

        # Replace the chain and readings
        beamer_bot.chain = mock_chain
        beamer_bot.readings = "Test readings"
        beamer_bot.prior_lesson = "Previous lesson content"

        slides = beamer_bot.generate_slides()

        # Assertions
        assert mock_slides_latex in slides
        mock_chain.invoke.assert_called_once()
        mock_validator.assert_called_once()
        mock_validate_latex.assert_called_once()

        # Verify invoke arguments
        expected_invoke_args = {
            "objectives": "Test objectives\n\nTest objectives\n\nTest objectives",
            "information": "Test readings",
            "last_presentation": "Previous lesson content",
            "lesson_no": 1,
            "specific_guidance": "Not provided.",
            "additional_guidance": ""
        }
        mock_chain.invoke.assert_called_once_with(expected_invoke_args)


def test_generate_prompt(beamer_bot):
    # Set specific attributes for controlled prompt generation
    beamer_bot.readings = "Sample readings"
    beamer_bot.prompt = beamer_bot._generate_prompt()

    assert "lesson 1" not in beamer_bot.prompt.messages[1].prompt.template
    assert "Sample readings" not in beamer_bot.prompt.messages[1].prompt.template  # should only show when calling chain.invoke or prompt.format()
    # Check that placeholders are present in HumanMessage (the second part of prompt messages)
    assert "{objectives}" in beamer_bot.prompt.messages[1].prompt.template
    assert "{information}" in beamer_bot.prompt.messages[1].prompt.template
    assert "{last_presentation}" in beamer_bot.prompt.messages[1].prompt.template


@patch.object(BeamerBot, '_validate_llm_response')
@patch('class_factory.beamer_bot.BeamerBot.validate_latex')
@patch('class_factory.utils.load_documents.LessonLoader.load_beamer_presentation')
@patch('class_factory.utils.load_documents.LessonLoader.extract_lesson_objectives')
def test_generate_slides_retries(
    mock_extract_obj,
    mock_load_beamer,
    mock_validate_latex,
    mock_validator,
    beamer_bot,
    caplog
):
    """Test retry logic in generate_slides method."""
    # Setup basic mocks
    mock_load_beamer.return_value = "Previous lesson content"
    mock_extract_obj.return_value = "Test objectives"

    # Create a mock for the LLM output that's consistent for all retries
    mock_slides_latex = "\\title{Retry Lesson}\n\\author{Retry Author}\n\\institute{Retry Institute}\n\\date{2024-01-02}\n\\begin{document}\n...slides...\\end{document}"

    # Create a chainable mock for the LLM
    mock_chain = MagicMock()

    # Use a MagicMock for the slides_data with a to_latex method
    mock_slides = MagicMock()
    mock_slides.to_latex.return_value = mock_slides_latex

    # Always return our mock slides object
    mock_chain.invoke.return_value = mock_slides

    # Assign to beamer_bot
    beamer_bot.chain = mock_chain

    # Configure validation side effects for retry
    mock_validator.side_effect = [
        {"status": 0, "additional_guidance": "Try improving structure."},  # First call fails
        {"status": 1, "additional_guidance": ""},  # Second call passes
        {"status": 1, "additional_guidance": ""}   # Third call passes
    ]

    # Configure LaTeX validation to fail then pass
    mock_validate_latex.side_effect = [False, True]

    # Run generate_slides and check retries
    with caplog.at_level(logging.WARNING):
        slides = beamer_bot.generate_slides()

    # Assert that the validator was called three times due to the retry logic
    assert mock_validator.call_count == 3
    assert mock_validate_latex.call_count == 2
    mock_chain.invoke.assert_called()  # Ensure `chain.invoke` was called
    assert mock_slides_latex in slides  # Check that generated content is in the output

    # Check logs for expected warning messages
    assert any("Response validation failed on attempt 1" in record.message for record in caplog.records)
    assert any("LaTeX code is invalid. Attempting a second model run." in record.message for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__])
