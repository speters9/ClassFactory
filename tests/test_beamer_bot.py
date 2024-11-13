from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from class_factory.beamer_bot.BeamerBot import BeamerBot  # Updated import path


@pytest.fixture
def mock_llm():
    return Mock()


@pytest.fixture
def mock_paths():
    # Define mock paths without creating directories
    syllabus_path = Path("/mocked/path/syllabus.txt")
    reading_dir = Path("/mocked/path/readings")
    slide_dir = Path("/mocked/path/slides")
    output_dir = Path("/mocked/path/output")

    return {
        "syllabus_path": syllabus_path,
        "reading_dir": reading_dir,
        "slide_dir": slide_dir,
        "output_dir": output_dir
    }


@pytest.fixture
def beamer_bot(mock_llm, mock_paths):
    with patch('class_factory.beamer_bot.BeamerBot.BeamerBot._validate_dir_path', side_effect=lambda path, name, must_contain_contents=False: Path(path)), \
            patch('class_factory.beamer_bot.BeamerBot.BeamerBot._validate_file_path', return_value=Path("/mocked/path/syllabus.txt")), \
            patch('pathlib.Path.is_file', return_value=True):
        return BeamerBot(
            lesson_no=1,
            syllabus_path=mock_paths["syllabus_path"],
            reading_dir=mock_paths["reading_dir"],
            slide_dir=mock_paths["slide_dir"],
            llm=mock_llm,
            output_dir=mock_paths["output_dir"],
            course_name="American Government"
        )


def test_beamer_bot_initialization(beamer_bot, mock_paths):
    """Test BeamerBot initialization with the fixture."""
    assert beamer_bot.lesson_no == 1
    assert beamer_bot.syllabus_path == mock_paths["syllabus_path"]
    assert beamer_bot.reading_dir == mock_paths["reading_dir"]
    assert beamer_bot.slide_dir == mock_paths["slide_dir"]
    assert beamer_bot.output_dir == mock_paths["output_dir"]
    assert isinstance(beamer_bot.llm, Mock)
    assert beamer_bot.course_name == "American Government"


@patch('class_factory.beamer_bot.BeamerBot.load_lessons', return_value=["Reading 1", "Reading 2"])
def test_load_readings(mock_load_lessons, beamer_bot):
    """Test loading readings using the fixture."""
    # Call the `_load_readings` method
    readings = beamer_bot._load_readings()

    # Verify that `load_lessons` was called with the expected arguments
    expected_input_dir = beamer_bot.reading_dir / f'L{beamer_bot.lesson_no}'
    mock_load_lessons.assert_called_once_with(expected_input_dir, lesson_range=1, recursive=False)

    # Check that the readings returned match the expected output
    assert readings == "Reading 1\n\nReading 2"


@patch('class_factory.beamer_bot.BeamerBot.extract_lesson_objectives')
@patch('class_factory.beamer_bot.BeamerBot.load_beamer_presentation')
@patch('class_factory.beamer_bot.BeamerBot.validate_latex')
@patch('class_factory.beamer_bot.BeamerBot.clean_latex_content')
def test_generate_slides(
    mock_clean_latex,
    mock_validate_latex,
    mock_load_beamer,
    mock_extract_objectives,
    beamer_bot
):
    """Test slide generation using the fixture."""
    # Set up return values
    mock_extract_objectives.return_value = "Test objectives"
    mock_load_beamer.return_value = "Previous lesson content"
    mock_validate_latex.return_value = True
    mock_clean_latex.return_value = "Cleaned LaTeX content"

    # Create a mock chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Generated LaTeX content"

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

        slides = beamer_bot.generate_slides()

        # Assertions
        assert "Cleaned LaTeX content" in slides
        mock_chain.invoke.assert_called_once()
        mock_validator.assert_called_once()
        mock_validate_latex.assert_called_once()

        # Verify invoke arguments
        expected_invoke_args = {
            "objectives": "Test objectives",
            "information": "Test readings",
            "last_presentation": "Previous lesson content",
            "specific_guidance": "Not provided.",
            "additional_guidance": ""
        }
        mock_chain.invoke.assert_called_once_with(expected_invoke_args)


@patch('builtins.open', new_callable=mock_open)
def test_save_slides(mock_open, beamer_bot):
    test_content = "Test LaTeX content"
    beamer_bot.save_slides(test_content)

    mock_open.assert_called_once_with(beamer_bot.beamer_output, 'w', encoding='utf-8')
    mock_open().write.assert_called_once_with(test_content)


if __name__ == "__main__":
    pytest.main([__file__])
