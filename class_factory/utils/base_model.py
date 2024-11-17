import logging
from pathlib import Path
from typing import Dict, List, Union

from pyprojroot.here import here

from class_factory.utils.load_documents import LessonLoader


class BaseModel:
    def __init__(self, lesson_no: int, course_name: str, lesson_loader: LessonLoader, output_dir: Path = None, verbose: bool = False):
        self.lesson_no = lesson_no
        self.course_name = course_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Initialize LessonLoader if not provided
        self.lesson_loader = lesson_loader

        # Paths and files
        self.output_dir = output_dir or here() / "ClassFactoryOutput"
        self.user_objectives = None

    def _load_readings(self, lesson_numbers: Union[int, range]) -> Dict[str, List[str]]:
        """
        Auto-loads readings for each lesson number provided and returns a dictionary with lesson numbers as keys.

        Args:
            lesson_numbers (Union[int, range]): A single lesson number or range of lesson numbers to load.

        Returns:
            Dict[str, List[str]]: A dictionary where each key is a lesson number and each value is a list of readings.
        """
        # Ensure lesson_numbers is always a range
        if isinstance(lesson_numbers, int):
            lesson_numbers = range(lesson_numbers, lesson_numbers + 1)

        # Directly pass the range to `LessonLoader` for reading retrieval
        return self.lesson_loader.load_lessons(lesson_number_or_range=lesson_numbers)

    def _set_user_objectives(self, objectives: Union[List[str], Dict[str, str]], lesson_range: Union[int, range]):
        """
        Set the lesson objectives provided by the user.

        If a list is provided, it is converted to a dictionary with keys in the format 'Lesson X',
        where 'X' corresponds to each lesson in the `lesson_range`. If a dictionary is provided, it
        should already be structured with lesson numbers as keys.

        Args:
            objectives (Union[List[str], Dict[str, str]]): The user-provided lesson objectives, either as a list
                (which will be converted to a dictionary) or as a dictionary where keys correspond to the corresponding lesson indicator.
            lesson_range (Union[int, range]): Range or single integer representing the lessons for which objectives are set.

        Raises:
            ValueError: If the length of the objectives list does not match the number of lessons in `lesson_range`.
            TypeError: If `objectives` is not a list or dictionary.
        """
        # Convert lesson_range to a range object if it is an int
        if isinstance(lesson_range, int):
            lesson_range = range(lesson_range, lesson_range + 1)

        if isinstance(objectives, list):
            if len(objectives) != len(lesson_range):
                raise ValueError("Length of objectives list must match the number of lessons in lesson_range.")
            self.user_objectives = {f'Lesson {i}': obj for i, obj in zip(lesson_range, objectives)}
        elif isinstance(objectives, dict):
            if len(objectives) != len(lesson_range):
                raise ValueError("Length of objectives list must match the number of lessons in lesson_range.")
            self.user_objectives = objectives
        else:
            raise TypeError("Objectives must be provided as either a list or a dictionary.")


if __name__ == "__main__":
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    # Load environment variables
    user_home = Path.home()
    load_dotenv()

    # Load paths from environment variables
    slide_dir = user_home / os.getenv('slideDir')
    syllabus_path = user_home / os.getenv('syllabus_path')
    readings_dir = user_home / os.getenv('readingsDir')

    # Define the lesson number for testing
    lesson_no = 8
    lesson_range = range(8, 10)  # Example range for multiple lessons

    # Initialize LessonLoader and BaseModel
    loader = LessonLoader(syllabus_path=syllabus_path, reading_dir=readings_dir, slide_dir=slide_dir)
    test_model = BaseModel(lesson_no=lesson_no, course_name="Sample Course", lesson_loader=loader, output_dir=slide_dir)

    # Test loading readings
    print("Testing _load_readings:")
    readings = test_model._load_readings(lesson_range)
    print("Readings Loaded:", readings)

    # Test setting user objectives
    print("\nTesting _set_user_objectives:")
    sample_objectives_list = ["Understand fundamentals", "Explore advanced topics"]
    test_model._set_user_objectives(sample_objectives_list, lesson_range)
    print("User Objectives Set (from list):", test_model.user_objectives)

    # Test setting user objectives with a dictionary
    sample_objectives_dict = {
        "Lesson 8": "Understand fundamentals",
        "Lesson 9": "Explore advanced topics"
    }
    test_model._set_user_objectives(sample_objectives_dict, lesson_range)
    print("User Objectives Set (from dict):", test_model.user_objectives)
