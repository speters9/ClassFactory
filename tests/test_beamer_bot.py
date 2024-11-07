from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from class_factory.beamer_bot.BeamerBot import BeamerBot  # Updated import path


@pytest.fixture
def mock_llm():
    return Mock()


@pytest.fixture
def mock_paths(tmp_path):
    syllabus_path = tmp_path / "syllabus.txt"
    reading_dir = tmp_path / "readings"
    slide_dir = tmp_path / "slides"
    output_dir = tmp_path / "output"

    # Create necessary directories and files
    reading_dir.mkdir(parents=True)
    slide_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (reading_dir / "L1").mkdir()
    (reading_dir / "L1" / "sample_reading.txt").write_text("Sample reading content")
    (slide_dir / "L0.tex").write_text("Previous lesson content")
    syllabus_path.write_text("Lesson 1 Objectives: Test objective")

    return {
        "syllabus_path": syllabus_path,
        "reading_dir": reading_dir,
        "slide_dir": slide_dir,
        "output_dir": output_dir
    }


@patch('class_factory.beamer_bot.BeamerBot.verify_lesson_dir', return_value=True)
@patch('class_factory.beamer_bot.BeamerBot.verify_beamer_file', return_value=True)
def test_beamer_bot_initialization(mock_verify_beamer, mock_verify_lesson, mock_llm, mock_paths):
    bot = BeamerBot(
        lesson_no=1,
        syllabus_path=mock_paths["syllabus_path"],
        reading_dir=mock_paths["reading_dir"],
        slide_dir=mock_paths["slide_dir"],
        llm=mock_llm,
        output_dir=mock_paths["output_dir"],
        course_name="American Government"
    )

    assert bot.lesson_no == 1
    assert bot.syllabus_path == mock_paths["syllabus_path"]
    assert bot.reading_dir == mock_paths["reading_dir"]
    assert bot.slide_dir == mock_paths["slide_dir"]
    assert bot.output_dir == mock_paths["output_dir"]
    assert bot.llm == mock_llm


@patch('class_factory.beamer_bot.BeamerBot.load_lessons')
@patch('class_factory.beamer_bot.BeamerBot.verify_lesson_dir', return_value=True)
@patch('class_factory.beamer_bot.BeamerBot.verify_beamer_file', return_value=True)
def test_load_readings(mock_verify_beamer, mock_verify_lesson, mock_load_lessons, mock_llm, mock_paths):
    mock_load_lessons.return_value = ["Reading 1", "Reading 2"]

    bot = BeamerBot(
        lesson_no=1,
        syllabus_path=mock_paths["syllabus_path"],
        reading_dir=mock_paths["reading_dir"],
        slide_dir=mock_paths["slide_dir"],
        llm=mock_llm,
        course_name="American Government"
    )

    mock_load_lessons.reset_mock()  # Reset the mock after initialization

    readings = bot._load_readings()
    assert readings == "Reading 1\n\nReading 2"
    mock_load_lessons.assert_called_once_with(bot.input_dir, lesson_range=1, recursive=False)


@patch('class_factory.beamer_bot.BeamerBot.extract_lesson_objectives')
@patch('class_factory.beamer_bot.BeamerBot.load_beamer_presentation')
@patch('class_factory.beamer_bot.BeamerBot.verify_lesson_dir', return_value=True)
@patch('class_factory.beamer_bot.BeamerBot.verify_beamer_file', return_value=True)
@patch('class_factory.beamer_bot.BeamerBot.validate_latex', return_value=True)
@patch('class_factory.beamer_bot.BeamerBot.clean_latex_content', return_value="Cleaned LaTeX content")
@patch('langchain_core.prompts.PromptTemplate.from_template')
@patch('langchain_core.output_parsers.StrOutputParser')  # For the main LaTeX content chain
def test_generate_slides(mock_str_parser, mock_prompt_template, mock_clean_latex, mock_validate_latex,
                         mock_verify_beamer, mock_verify_lesson, mock_load_beamer,
                         mock_extract_objectives, mock_llm, mock_paths):
    # Mock responses for lesson objectives and prior presentation content
    mock_extract_objectives.return_value = "Test objectives"
    mock_load_beamer.return_value = "Previous lesson content"

    # Mock the main chain (LLM) to return LaTeX content as a string
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Generated LaTeX content"  # Simulated LaTeX response from main chain

    # Mock the chain creation process for the main LLM chain
    mock_prompt_template.return_value = MagicMock()
    mock_prompt_template.return_value.__or__.return_value = MagicMock()
    mock_prompt_template.return_value.__or__.return_value.__or__.return_value = mock_chain

    # Mock the validation response separately to simulate the validator chain
    with patch.object(BeamerBot, '_validate_llm_response') as mock_validator:
        mock_validator.return_value = {
            "evaluation_score": 8.5,
            "status": 1,
            "reasoning": "Slides are accurate and well-structured.",
            "additional_guidance": ""
        }  # Validator response with JSON structure

        # Instantiate BeamerBot with mocked paths and LLM
        bot = BeamerBot(
            lesson_no=1,
            syllabus_path=mock_paths["syllabus_path"],
            reading_dir=mock_paths["reading_dir"],
            slide_dir=mock_paths["slide_dir"],
            llm=mock_llm,
            course_name="American Government"
        )

        # Mock `load_readings` to avoid file dependency
        with patch.object(bot, '_load_readings', return_value="Test readings"):
            slides = bot.generate_slides()

    # Assertions to verify expected behavior
    assert "Cleaned LaTeX content" in slides  # Ensure final LaTeX content includes cleaned content
    mock_chain.invoke.assert_called_once()  # Ensure main LLM chain was called
    mock_validator.assert_called_once()  # Validator should be invoked once
    mock_validate_latex.assert_called_once()  # Ensure LaTeX validation step was called
    mock_clean_latex.assert_called_once_with("Generated LaTeX content")  # Ensure LaTeX cleaning was done


@patch('class_factory.beamer_bot.BeamerBot.verify_lesson_dir', return_value=True)
@patch('class_factory.beamer_bot.BeamerBot.verify_beamer_file', return_value=True)
def test_save_slides(mock_verify_beamer, mock_verify_lesson, mock_llm, mock_paths):
    bot = BeamerBot(
        lesson_no=1,
        syllabus_path=mock_paths["syllabus_path"],
        reading_dir=mock_paths["reading_dir"],
        slide_dir=mock_paths["slide_dir"],
        llm=mock_llm,
        output_dir=mock_paths["output_dir"],
        course_name="American Government"
    )

    test_content = "Test LaTeX content"
    bot.save_slides(test_content)

    saved_file = mock_paths["output_dir"] / "L1.tex"
    assert saved_file.exists()
    assert saved_file.read_text() == test_content


if __name__ == "__main__":
    pytest.main([__file__])
