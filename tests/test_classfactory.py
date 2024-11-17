import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from class_factory.ClassFactory import ClassFactory
from class_factory.utils.load_documents import LessonLoader

sys.modules["gradio"] = MagicMock()
sys.modules["gradio.themes"] = MagicMock()
sys.modules["gradio.components"] = MagicMock()


@pytest.fixture
def mock_paths():
    # Create temporary directories
    temp_dirs = {
        "syllabus_path": Path(tempfile.mkdtemp()) / "syllabus.txt",
        "reading_dir": Path(tempfile.mkdtemp()),
        "slide_dir": Path(tempfile.mkdtemp()),
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
def factory(mock_paths):
    """Fixture to initialize the ClassFactory object with mocked paths and llm."""
    lesson_no = 1
    syllabus_path = mock_paths["syllabus_path"]
    reading_dir = mock_paths["reading_dir"]
    slide_dir = mock_paths["slide_dir"]
    project_dir = mock_paths["project_dir"]
    output_dir = mock_paths["output_dir"]

    llm = MagicMock()  # Mock the llm object
    with patch('pathlib.Path.is_file', return_value=True):
        return ClassFactory(
            lesson_no=lesson_no,
            syllabus_path=syllabus_path,
            reading_dir=reading_dir,
            project_dir=project_dir,
            output_dir=None,
            llm=llm,
            lesson_range=range(1, 2),
            verbose=False
        )


# @pytest.fixture
# def patch_validations():
#     with patch('class_factory.ClassFactory.ClassFactory.lesson_loader._validate_file_path', return_value=Path("mocked_syllabus.docx")), \
#          patch('class_factory.ClassFactory.ClassFactory.lesson_loader._validate_dir_path', side_effect=lambda path, name, must_contain_contents=False: Path(path)), \
#          patch('class_factory.beamer_bot.BeamerBot.BeamerBot.lesson_loader._validate_file_path', return_value=Path("mocked_syllabus.docx")), \
#          patch('class_factory.beamer_bot.BeamerBot.BeamerBot.lesson_loader._validate_dir_path', side_effect=lambda path, name, must_contain_contents=False: Path(path)), \
#          patch('class_factory.concept_web.ConceptWeb.ConceptMapBuilder.lesson_loader._validate_file_path', return_value=Path("mocked_syllabus.docx")), \
#          patch('class_factory.concept_web.ConceptWeb.ConceptMapBuilder.lesson_loader._validate_dir_path', side_effect=lambda path, name, must_contain_contents=False: Path(path)), \
#          patch('class_factory.quiz_maker.QuizMaker.QuizMaker.lesson_loader._validate_file_path', return_value=Path("mocked_syllabus.docx")), \
#          patch('class_factory.quiz_maker.QuizMaker.QuizMaker.lesson_loader._validate_dir_path', side_effect=lambda path, name, must_contain_contents=False: Path(path)), \
#          patch('pathlib.Path.is_file', return_value=True):
#         yield

@pytest.fixture
def patch_validations():
    with patch('class_factory.utils.load_documents.LessonLoader._validate_file_path', return_value=Path("mocked_syllabus.docx")), \
            patch('class_factory.utils.load_documents.LessonLoader._validate_dir_path', side_effect=lambda path, name, must_contain_contents=False: Path(path)), \
            patch('pathlib.Path.is_file', return_value=True):
        yield


def test_classfactory_initialization(factory):
    """Test that ClassFactory initializes correctly with a lesson_loader and other attributes."""

    # Verify primary ClassFactory attributes
    assert factory.lesson_no == 1
    assert factory.lesson_range == range(1, 2)
    assert factory.course_name == "Political Science"
    assert factory.llm is not None  # Ensure llm is initialized

    # Verify output_dir is correctly set
    assert "ClassFactoryOutput" in str(factory.output_dir)

    # Verify lesson_loader attributes
    assert factory.lesson_loader is not None
    assert isinstance(factory.lesson_loader, LessonLoader)
    assert factory.lesson_loader.syllabus_path.exists()
    assert factory.lesson_loader.reading_dir.exists()
    assert factory.lesson_loader.project_dir.exists()


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
@patch('pathlib.Path.is_file', return_value=True)
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
