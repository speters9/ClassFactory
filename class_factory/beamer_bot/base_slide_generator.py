"""
**BaseSlideGenerator Module**
------------------------------

Provides an abstract base class for all slide generators (LaTeX, PowerPoint, etc.).
This class extracts common functionality shared across different output formats while
allowing format-specific implementations to handle their unique requirements.

Key Functionalities
~~~~~~~~~~~~~~~~~~~

1. **Common Setup**:
   - Lesson reading and objective loading
   - LLM initialization and configuration
   - Output directory management
   - Validation framework setup

2. **Abstract Methods**:
   - Slide generation (format-specific)
   - Slide saving (format-specific)
   - Prompt generation (format-specific)

3. **Shared Utilities**:
   - Reading formatting for prompts
   - Prior lesson retrieval
   - Objective formatting
   - User objective management

Usage
~~~~~

This is an abstract base class and should not be instantiated directly.
Instead, create concrete implementations like LatexSlideGenerator or PptxSlideGenerator.

.. code-block:: python

    class LatexSlideGenerator(BaseSlideGenerator):
        def generate_slides(self, specific_guidance=None):
            # LaTeX-specific implementation
            pass

        def save_slides(self, content):
            # LaTeX-specific saving
            pass
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

from class_factory.utils.base_model import BaseModel
from class_factory.utils.llm_validator import Validator
from class_factory.utils.load_documents import LessonLoader


class BaseSlideGenerator(BaseModel, ABC):
    """
    Abstract base class for slide generators across different output formats.

    This class provides common functionality for all slide generators while requiring
    subclasses to implement format-specific generation and saving logic.

    Attributes:
        lesson_no (int): Lesson number for which to generate slides.
        llm: Language model instance for generating slides.
        course_name (str): Name of the course for slide context.
        lesson_loader (LessonLoader): Loader for accessing lesson readings and objectives.
        output_dir (Path): Directory to save the generated slides.
        slide_dir (Path): Directory containing existing slides (for prior lesson reference).
        validator (Validator): Validator instance for content quality checking.
        readings (str): Formatted readings for use in prompts.
        prior_lesson (int): Number of the most recent prior lesson found.

    Abstract Methods:
        generate_slides: Generate slides in the specific format
        save_slides: Save slides in the specific format
        _generate_prompt: Create format-specific prompts for the LLM
    """

    def __init__(
        self,
        lesson_no: int,
        llm,
        course_name: str,
        lesson_loader: LessonLoader,
        output_dir: Union[Path, str] = None,
        verbose: bool = False,
        slide_dir: Union[Path, str] = None,
        lesson_objectives: dict = None
    ):
        """
        Initialize the base slide generator with common configuration.

        Args:
            lesson_no (int): Lesson number for which to generate slides
            llm: Language model instance for content generation
            course_name (str): Name of the course
            lesson_loader (LessonLoader): Loader for lesson materials
            output_dir (Union[Path, str], optional): Output directory for generated slides
            verbose (bool, optional): Enable verbose logging. Defaults to False.
            slide_dir (Union[Path, str], optional): Directory with existing slides
            lesson_objectives (dict, optional): User-provided lesson objectives
        """
        super().__init__(
            lesson_no=lesson_no,
            course_name=course_name,
            lesson_loader=lesson_loader,
            output_dir=output_dir,
            verbose=verbose
        )

        self.llm = llm
        self.llm_response = None

        # Determine slide directory
        if slide_dir:
            self.lesson_loader.slide_dir = slide_dir
            self.slide_dir = self.lesson_loader.slide_dir
        elif not slide_dir and not self.lesson_loader.slide_dir:
            self.logger.warning(
                "No slide directory provided directly or through lesson loader. "
                "Some functionality, such as loading prior presentations, may be limited."
            )

        # Format readings for prompts
        self.readings = self._format_readings_for_prompt()

        # Set user objectives if provided
        self.user_objectives = (
            self.set_user_objectives(lesson_objectives, range(self.lesson_no, self.lesson_no + 1))
            if lesson_objectives else {}
        )

        # Initialize validator
        self.validator = Validator(llm=self.llm, log_level=self.logger.level)

        # Track prior lesson number
        self.prior_lesson = self.lesson_no - 1  # Updated when prior presentation is found

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _format_readings_for_prompt(self) -> str:
        """
        Format readings as a single string for use in the LLM prompt.

        Returns:
            str: Combined readings across all specified lessons for the LLM prompt.
        """
        all_readings_dict = self._load_readings(self.lesson_no)
        combined_readings = []

        for lesson, readings in all_readings_dict.items():
            for idx, reading in enumerate(readings, start=1):
                combined_readings.append(
                    f"Lesson {lesson}, Reading {idx}:\n{reading}\n"
                )

        return "\n".join(combined_readings)

    def _find_prior_lesson(self, lesson_no: int, max_attempts: int = 3, file_extension: str = ".tex") -> Path:
        """
        Find the most recent prior lesson's slide file to use as a template.

        Args:
            lesson_no (int): The current lesson number.
            max_attempts (int): Number of prior lessons to attempt to retrieve. Defaults to 3.
            file_extension (str): File extension to search for. Defaults to ".tex".

        Returns:
            Path: The path to the located slide file.

        Raises:
            FileNotFoundError: If no valid prior lesson file is found within max_attempts.
        """
        for i in range(1, max_attempts + 1):
            prior_lesson = lesson_no - i
            slide_file = self.lesson_loader.slide_dir / f'L{prior_lesson}{file_extension}'

            # Check if the slide file exists for this prior lesson
            if slide_file.is_file():
                self.prior_lesson = int(prior_lesson)
                self.logger.info(f"Found prior lesson: Lesson {prior_lesson}")
                return slide_file

        # Raise error if no valid prior slide file is found within the attempts
        raise FileNotFoundError(
            f"No prior slide file found within the last {max_attempts} lessons."
        )

    @abstractmethod
    def generate_slides(self, specific_guidance: str = None, lesson_objectives: dict = None, **kwargs) -> Any:
        """
        Generate slides for the lesson using the language model.

        This method must be implemented by subclasses to handle format-specific
        slide generation logic.

        Args:
            specific_guidance (str, optional): Custom instructions for slide content and structure
            lesson_objectives (dict, optional): Override default objectives with custom ones
            **kwargs: Additional format-specific arguments

        Returns:
            Any: Generated slides in the appropriate format (str for LaTeX, object for PowerPoint, etc.)

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement generate_slides()")

    @abstractmethod
    def save_slides(self, content: Any, output_dir: Union[Path, str] = None) -> None:
        """
        Save the generated slides to a file.

        This method must be implemented by subclasses to handle format-specific
        saving logic.

        Args:
            content (Any): The slides content to save (format depends on generator type)
            output_dir (Union[Path, str], optional): Override default output directory

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement save_slides()")

    @abstractmethod
    def _generate_prompt(self, human_prompt: str = None):
        """
        Generate format-specific prompts for the LLM.

        This method must be implemented by subclasses to create prompts appropriate
        for their output format (LaTeX syntax, PowerPoint structure, etc.).

        Args:
            human_prompt (str, optional): Override default human prompt template

        Returns:
            The constructed prompt object (format varies by implementation)

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement _generate_prompt()")
