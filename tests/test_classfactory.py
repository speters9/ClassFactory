from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from class_factory.ClassFactory import ClassFactory


@pytest.fixture
def factory():
    """Fixture to initialize the ClassFactory object with mocked paths and llm."""
    lesson_no = 21
    syllabus_path = Path('fake_syllabus.docx')
    reading_dir = Path('fake_reading_dir')
    slide_dir = Path('fake_slide_dir')
    llm = MagicMock()  # Mock the llm object
    return ClassFactory(
        lesson_no=lesson_no,
        syllabus_path=syllabus_path,
        reading_dir=reading_dir,
        llm=llm,
        lesson_range=range(21, 22)
    )


@pytest.fixture
def patch_validations():
    with patch('class_factory.beamer_bot.BeamerBot.BeamerBot._validate_file_path', return_value=Path("mocked_syllabus.docx")), \
            patch('class_factory.beamer_bot.BeamerBot.BeamerBot._validate_dir_path', side_effect=lambda path, name, must_contain_contents=False: Path(path)), \
            patch('pathlib.Path.is_file', return_value=True), \
            patch('class_factory.concept_web.ConceptWeb.ConceptMapBuilder._validate_file_path', return_value=Path("mocked_syllabus.docx")), \
            patch('class_factory.concept_web.ConceptWeb.ConceptMapBuilder._validate_dir_path', side_effect=lambda path, name, must_contain_contents=False: Path(path)), \
            patch('class_factory.quiz_maker.QuizMaker.QuizMaker._validate_file_path', return_value=Path("mocked_syllabus.docx")), \
            patch('class_factory.quiz_maker.QuizMaker.QuizMaker._validate_dir_path', side_effect=lambda path, name, must_contain_contents=False: Path(path)):
        yield


@patch('class_factory.beamer_bot.BeamerBot.BeamerBot', new=MagicMock())
def test_create_beamerbot(factory, patch_validations):
    """Test creating BeamerBot module with patched validations and mocked BeamerBot."""
    slide_dir = Path('fake_slide_dir')
    beamerbot = factory.create_module(
        'BeamerBot',
        verbose=True,
        slide_dir=slide_dir,
        course_name="Political Science"
    )
    assert isinstance(beamerbot, MagicMock)


# Test for creating ConceptWeb module


@patch('class_factory.concept_web.ConceptWeb.ConceptMapBuilder', new=MagicMock())
def test_create_conceptweb(factory, patch_validations):
    """Test creating ConceptWeb module with patched validations and mocked ConceptWeb."""
    conceptweb = factory.create_module(
        'ConceptWeb',
        lesson_range=range(17, 21),
        verbose=False
    )
    assert isinstance(conceptweb, MagicMock)


@patch('class_factory.quiz_maker.QuizMaker.QuizMaker', new=MagicMock())
def test_create_quizmaker(factory, patch_validations):
    """Test creating QuizMaker module with patched validations and mocked QuizMaker."""
    quizmaker = factory.create_module('QuizMaker', verbose=False)
    assert isinstance(quizmaker, MagicMock)


def test_create_module_invalid(factory):
    """Test creating a module with an invalid module name."""
    with pytest.raises(ValueError) as excinfo:
        factory.create_module('InvalidModule')
    assert "Module InvalidModule not recognized." in str(excinfo.value)


# Test for unrecognized module name


def test_create_module_invalid(factory):
    with pytest.raises(ValueError) as excinfo:
        factory.create_module('InvalidModule')
    assert "Module InvalidModule not recognized." in str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__])
