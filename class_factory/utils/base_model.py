import logging
from pathlib import Path
from typing import Dict, List, Union

from pyprojroot.here import here

from class_factory.utils.load_documents import LessonLoader


class BaseModel:
    """
    A base class for educational modules that provides common setup and utility functions, such as loading
    lesson readings and setting user-defined objectives.

    Attributes:
        lesson_no (int): The specific lesson number for the current instance.
        course_name (str): Name of the course, used as context in other methods and prompts.
        lesson_loader (LessonLoader): Instance for loading lesson-related data.
        output_dir (Path): Directory where outputs are saved; defaults to 'ClassFactoryOutput'.
        logger (Logger): Logger instance for the class.
        user_objectives (Optional[Dict[str, str]]): Dictionary of user-defined objectives, if provided.

    Methods:
        _load_readings(lesson_numbers: Union[int, range]) -> Dict[str, List[str]]:
            Loads and returns readings for the specified lesson(s) as a dictionary.

        set_user_objectives(objectives: Union[List[str], Dict[str, str]], lesson_range: Union[int, range]) -> Dict[str, str]:
            Sets user-defined objectives for each lesson in the specified range and updates self.user_objectives.

        _get_lesson_objectives(lesson_num: int) -> str:
            Retrieves lesson objectives for a given lesson number, falling back to extracted objectives if no user objectives exist.
    """

    def __init__(self, lesson_no: int, course_name: str, lesson_loader: LessonLoader, output_dir: Union[Path, str] = None, verbose: bool = False):
        """
        Initialize the BaseModel with essential attributes, paths, and logging configurations.

        Args:
            lesson_no (int): Lesson number for the current instance.
            course_name (str): Name of the course.
            lesson_loader (LessonLoader): An instance of the LessonLoader for loading lesson-related data.
            output_dir (Path, optional): Directory for saving outputs; defaults to 'ClassFactoryOutput'.
            verbose (bool, optional): If True, sets logging level to INFO; otherwise, WARNING.
        """
        self.lesson_no = lesson_no
        self.course_name = course_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Initialize LessonLoader if not provided
        self.lesson_loader = lesson_loader

        # Paths and files
        self.output_dir = Path(output_dir) if output_dir else here() / "ClassFactoryOutput"
        self.user_objectives = None

    def _load_readings(self, lesson_numbers: Union[int, range]) -> Dict[str, List[str]]:
        """
        Load readings for the specified lesson numbers, returning a dictionary of readings by lesson.

        Args:
            lesson_numbers (Union[int, range]): A single lesson number or range of lesson numbers.

        Returns:
            Dict[str, List[str]]: A dictionary with lesson numbers as keys and lists of readings as values.
        """
        # Ensure lesson_numbers is always a range
        if isinstance(lesson_numbers, int):
            lesson_numbers = range(lesson_numbers, lesson_numbers + 1)

        # Directly pass the range to `LessonLoader` for reading retrieval
        return self.lesson_loader.load_lessons(lesson_number_or_range=lesson_numbers)

    def set_user_objectives(self, objectives: Union[List[str], Dict[str, str]], lesson_range: Union[int, range]) -> Dict[str, str]:
        """
        Set user-defined objectives for each lesson in `lesson_range`, supporting both list and dictionary formats.
        Updates the self.user_objectives attribute with the processed objectives.

        Args:
            objectives (Union[List[str], Dict[str, str]]): User-provided objectives, either as a list (converted to
                a dictionary) or as a dictionary keyed by lesson numbers as strings.
            lesson_range (Union[int, range]): Single lesson number or range of lesson numbers for objectives.

        Returns:
            Dict[str, str]: Processed dictionary of user objectives keyed by lesson numbers as strings.

        Raises:
            ValueError: If the number of objectives in the list does not match the number of lessons in `lesson_range`.
            TypeError: If `objectives` is neither a list nor a dictionary.
        """
        # Convert lesson_range to a range object if it is an int
        if isinstance(lesson_range, int):
            lesson_range = range(lesson_range, lesson_range + 1)

        if isinstance(objectives, list):
            if len(objectives) != len(lesson_range):
                raise ValueError("Length of objectives list must match the number of lessons in lesson_range.")
            # Map objectives to lesson numbers as strings
            user_objectives = {str(lesson): obj for lesson, obj in zip(lesson_range, objectives)}
        elif isinstance(objectives, dict):
            user_objectives = {}
            for key, value in objectives.items():
                try:
                    # Try converting the key to an integer and back to a string
                    numeric_key = str(int(key))
                    user_objectives[numeric_key] = value
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid key '{key}'. All keys must be numeric (e.g., '1', 2).")
        else:
            raise TypeError("Objectives must be provided as either a list or a dictionary.")

        # Assign and logs
        self.logger.info(f"User objectives set: {user_objectives}")
        return user_objectives

    def _get_lesson_objectives(self, lesson_num: int) -> str:
        """
        Retrieve lesson objectives for a given lesson number, including context.

        First, checks user-defined objectives. If not found, falls back to the
        lesson loader's extracted objectives. Ensures the output is consistently
        formatted with "Lesson {lesson_no}" included.

        Args:
            lesson_num (int): The lesson number for which to retrieve objectives.

        Returns:
            str: Objectives text prefixed with "Lesson {lesson_num}" for context.
                If no objectives are found, returns "Lesson {lesson_num}\n\n(No objectives provided)".
        """
        # Check user-defined objectives
        objectives = self.user_objectives.get(str(lesson_num), '')
        if objectives:
            return f"Lesson {lesson_num}\n\n{objectives.strip()}"

        # Fallback to lesson loader objectives
        objectives = self.lesson_loader.extract_lesson_objectives(lesson_num, only_current=True)
        if objectives:
            return objectives.strip()

        # Return an empty string if neither is available
        return f"Lesson {lesson_num}\n\n(No objectives provided)"


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
    print("\nTesting set_user_objectives:")
    sample_objectives_list = ["Understand fundamentals", "Explore advanced topics"]
    test_model.set_user_objectives(sample_objectives_list, lesson_range)
    print("User Objectives Set (from list):", test_model.user_objectives)

    # Test setting user objectives with a dictionary
    sample_objectives_dict = {
        "Lesson 8": "Understand fundamentals",
        "Lesson 9": "Explore advanced topics"
    }
    test_model.set_user_objectives(sample_objectives_dict, lesson_range)
    print("User Objectives Set (from dict):", test_model.user_objectives)
