from pathlib import Path
from unittest.mock import MagicMock

import pytest

from class_factory.utils.base_model import BaseModel
from class_factory.utils.load_documents import LessonLoader


@pytest.fixture
def mock_lesson_loader():
    # Create a mock LessonLoader with a sample return for load_lessons
    loader = MagicMock(spec=LessonLoader)
    loader.load_lessons.return_value = {"Lesson 1": ["Reading 1", "Reading 2"]}
    return loader


@pytest.fixture
def base_model(mock_lesson_loader):
    # Initialize BaseModel instance with mock loader
    return BaseModel(lesson_no=1, course_name="Test Course", lesson_loader=mock_lesson_loader, output_dir=Path("/tmp"))


def test_initialization(base_model):
    # Check attributes after initialization
    assert base_model.lesson_no == 1
    assert base_model.course_name == "Test Course"
    assert base_model.output_dir == Path("/tmp")
    assert base_model.user_objectives is None
    assert base_model.logger is not None


def test_load_readings_single_lesson(base_model, mock_lesson_loader):
    # Test loading readings for a single lesson number
    readings = base_model._load_readings(lesson_numbers=1)
    mock_lesson_loader.load_lessons.assert_called_once_with(lesson_number_or_range=range(1, 2))
    assert readings == {"Lesson 1": ["Reading 1", "Reading 2"]}


def test_load_readings_lesson_range(base_model, mock_lesson_loader):
    # Test loading readings for a range of lessons
    readings = base_model._load_readings(lesson_numbers=range(1, 3))
    mock_lesson_loader.load_lessons.assert_called_with(lesson_number_or_range=range(1, 3))
    assert readings == {"Lesson 1": ["Reading 1", "Reading 2"]}


def test_set_user_objectives_with_list(base_model):
    # Test setting user objectives using a list
    obj_list = ["Understand topic A", "Understand topic B"]
    objectives = base_model.set_user_objectives(objectives=obj_list, lesson_range=range(1, 3))
    base_model.user_objectives = objectives
    assert base_model.user_objectives == {"1": "Understand topic A", "2": "Understand topic B"}


def test_set_user_objectives_with_dict(base_model):
    # Test setting user objectives using a dictionary
    objectives = {"1": "Understand topic A", "2": "Understand topic B"}
    set_objectives = base_model.set_user_objectives(objectives=objectives, lesson_range=range(1, 3))
    base_model.user_objectives = set_objectives
    assert base_model.user_objectives == objectives


def test_set_user_objectives_mismatched_list_length(base_model):
    # Test error raised when objectives list length doesn't match lesson range length
    objectives = ["Understand topic A"]
    with pytest.raises(ValueError, match="Length of objectives list must match the number of lessons in lesson_range"):
        base_model.set_user_objectives(objectives=objectives, lesson_range=range(1, 3))


def test_set_user_objectives_invalid_objective_type(base_model):
    # Test error raised when objectives is neither list nor dictionary
    objectives = "Understand topic A"
    with pytest.raises(TypeError, match="Objectives must be provided as either a list or a dictionary"):
        base_model.set_user_objectives(objectives=objectives, lesson_range=range(1, 3))


if __name__ == "__main__":
    pytest.main([__file__])
