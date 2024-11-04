from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from class_factory.ClassFactory import ClassFactory


@pytest.fixture
def factory():
    """Fixture to initialize the ClassFactory object."""
    lesson_no = 21
    syllabus_path = Path('fake_syllabus.docx')
    reading_dir = Path('fake_reading_dir')
    slide_dir = Path('fake_slide_dir')
    llm = MagicMock()  # Mock the llm object
    return ClassFactory(lesson_no=lesson_no,
                        syllabus_path=syllabus_path,
                        reading_dir=reading_dir,
                        llm=llm,
                        lesson_range=range(21, 22))


# Test for creating BeamerBot module
@patch('class_factory.ClassFactory.BeamerBot')
def test_create_beamerbot(mock_beamerbot, factory):
    slide_dir = Path('fake_slide_dir')
    beamerbot = factory.create_module('BeamerBot', verbose=True, slide_dir=slide_dir,
                                      course_name="Political Science")
    mock_beamerbot.assert_called_once_with(
        lesson_no=factory.lesson_no,
        syllabus_path=factory.syllabus_path,
        reading_dir=factory.reading_dir,
        slide_dir=slide_dir,
        llm=factory.llm,
        course_name="Political Science",
        output_dir=factory.output_dir / f"BeamerBot/L{factory.lesson_no}",
        verbose=True,
    )
    assert isinstance(beamerbot, MagicMock)  # Since BeamerBot is mocked

# Test for creating ConceptWeb module


@patch('class_factory.ClassFactory.ConceptMapBuilder')
def test_create_conceptweb(mock_conceptweb, factory):
    conceptweb = factory.create_module('ConceptWeb', lesson_range=range(17, 21), verbose=False)
    mock_conceptweb.assert_called_once_with(
        lesson_range=range(17, 21),
        readings_dir=factory.reading_dir,
        syllabus_path=factory.syllabus_path,
        llm=factory.llm,
        project_dir=factory.project_dir,
        course_name="Political Science",
        output_dir=factory.output_dir / f"ConceptWeb/L{factory.lesson_no}",
        verbose=False,
        recursive=True
    )
    assert isinstance(conceptweb, MagicMock)  # Since ConceptMapBuilder is mocked

# Test for creating QuizMaker module


@patch('class_factory.ClassFactory.QuizMaker')
def test_create_quizmaker(mock_quizmaker, factory):
    quizmaker = factory.create_module('QuizMaker', verbose=False)
    mock_quizmaker.assert_called_once_with(
        lesson_range=factory.lesson_range,
        llm=factory.llm,
        syllabus_path=factory.syllabus_path,
        reading_dir=factory.reading_dir,
        output_dir=factory.output_dir / f"QuizMaker/L{factory.lesson_no}",
        course_name="Political Science",
        prior_quiz_path=factory.project_dir / "data/quizzes",
        verbose=False
    )
    assert isinstance(quizmaker, MagicMock)  # Since QuizMaker is mocked

# Test for unrecognized module name


def test_create_module_invalid(factory):
    with pytest.raises(ValueError) as excinfo:
        factory.create_module('InvalidModule')
    assert "Module InvalidModule not recognized." in str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__])
