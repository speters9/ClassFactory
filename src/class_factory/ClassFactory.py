"""
**ClassFactory**

This module defines the `ClassFactory` class, which serves as a factory for integrating several AI-enabled education modules. These modules include:

- **BeamerBot**: Automates the creation of LaTeX Beamer slides for lessons.
- **ConceptWeb**: Generates concept maps from lesson materials using a language model.
- **QuizMaker**: Creates quizzes based on lesson objectives and readings, with options for hosting interactive quizzes and analyzing quiz results.

The script provides a unified interface for managing these modules, allowing users to generate slides, concept maps, and quizzes for specified lessons or lesson ranges.

Key functionalities include:

1. **Module Creation**: The `create_module` method dynamically creates and returns instances of BeamerBot, ConceptWeb, or QuizMaker based on the provided module name.
2. **Customizable Output**: Each module’s output is saved in specific directories, organized by lesson number and module type. Outputs can be customized using optional arguments.
3. **Language Model Integration**: Uses an LLM (e.g., GPT-4, LLaMA) for generating content, which is passed to each module for content creation.

Usage:

1. Initialize the `ClassFactory` with the paths to the syllabus, readings, and slides, along with the lesson number and LLM instance.
2. Use `create_module` to generate specific modules and call their respective methods to create slides, concept maps, or quizzes.
3. Optionally, customize the output directories or provide additional configurations for each module via keyword arguments.

Dependencies:

- `BeamerBot`: For slide generation.
- `ConceptWeb`: For concept map generation.
- `QuizMaker`: For quiz creation, hosting, and analysis.
- Utility functions for loading lessons, resetting loggers, and managing file paths.

This script can be run as a standalone module, allowing for manual testing or demonstration of the factory's capabilities.
"""

from pathlib import Path
from typing import Optional, Union

from pyprojroot.here import here

from src.beamer_bot.BeamerBot import BeamerBot
from src.concept_web.ConceptWeb import ConceptMapBuilder
from src.quiz_maker.QuizMaker import QuizMaker
from src.utils.tools import reset_loggers

reset_loggers()


class ClassFactory:
    """
    A factory class responsible for creating and managing different educational modules.

    This class provides a unified interface to create instances of the following three modules:

    - **BeamerBot**: Automated LaTeX Beamer slide generation.
    - **ConceptWeb**: Automated concept map generation.
    - **QuizMaker**: Automated quiz generation, hosting, and analysis.

    Attributes:
        lesson_no (int): The lesson number for which modules are created.
        syllabus_path (Path): Path to the syllabus file.
        reading_dir (Path): Path to the directory containing lesson readings.
        slide_dir (Path): Path to the directory containing lesson slides.
        llm: The language model used for generating content in some modules.
        project_dir (Path): The base project directory.
        output_dir (Path): Directory where output from modules will be saved.
        lesson_range (range): Range of lessons to cover.

    By default, all module outputs will be saved in the root directory, under a directory titled "ClassFactoryOutput".
    """

    def __init__(self, lesson_no: int, syllabus_path: Union[str, Path], reading_dir: Union[str, Path],
                 slide_dir: Union[str, Path], llm, project_dir: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None, lesson_range: Optional[range] = None, **kwargs):
        """
        Initialize the ClassFactory with the necessary paths and configurations.

        Args:
            lesson_no (int): The lesson number for which to create modules.
            syllabus_path (Union[str, Path]): The path to the syllabus file.
            reading_dir (Union[str, Path]): The path to the directory with lesson readings.
            slide_dir (Union[str, Path]): The path to the directory with slides.
            llm: The language model used for generating content in some modules.
            project_dir (Optional[Union[str, Path]]): The base project directory. Defaults to current directory.
            output_dir (Optional[Union[str, Path]]): The directory where output will be saved. Defaults to 'ClassFactoryOutput'.
            lesson_range (Optional[range]): The range of lessons to be covered. Defaults to the lesson_no.
            **kwargs: Additional keyword arguments.
        """
        self.lesson_no = lesson_no
        self.syllabus_path = syllabus_path
        self.reading_dir = reading_dir
        self.slide_dir = slide_dir
        self.llm = llm
        self.project_dir = Path(project_dir) if project_dir else here()
        self.output_dir = Path(output_dir) if output_dir else here() / "ClassFactoryOutput"
        self.lesson_range = lesson_range if lesson_range else range(lesson_no, lesson_no + 1)  # Default to a single lesson

    def create_module(self, module_name: str, **kwargs) -> Union[BeamerBot, ConceptMapBuilder, QuizMaker]:
        """
        Create a specific module instance based on the provided module name.

        Args:
            module_name (str): The name of the module to create. Options are 'BeamerBot', 'ConceptWeb', or 'QuizMaker'.
            **kwargs: Additional keyword arguments for the specific module.

        Returns:
            Union[BeamerBot, ConceptMapBuilder, QuizMaker]: The created module instance.

        Raises:
            ValueError: If an invalid module name is provided.

        By default, each module output will be saved in its respective directory in the ClassFactoryOutput/ directory,
        in a directory named after the designated lesson number.
        """
        interim_output_dir = kwargs.get("output_dir", self.output_dir)
        if module_name in ["BeamerBot", "beamerbot"]:
            beamer_output_dir = interim_output_dir / f"BeamerBot/L{self.lesson_no}"
            beamer_output_dir.mkdir(parents=True, exist_ok=True)
            # BeamerBot should still use a single lesson (lesson_no)
            return BeamerBot(
                lesson_no=self.lesson_no,
                syllabus_path=self.syllabus_path,
                reading_dir=self.reading_dir,
                slide_dir=self.slide_dir,
                llm=self.llm,
                output_dir=beamer_output_dir,
                verbose=kwargs.get("verbose", False),
            )
        elif module_name in ["ConceptWeb", "conceptweb"]:
            concept_output_dir = interim_output_dir / f"ConceptWeb/L{self.lesson_no}"
            concept_output_dir.mkdir(parents=True, exist_ok=True)
            # ConceptMapBuilder uses the lesson range
            return ConceptMapBuilder(
                lesson_range=kwargs.get("lesson_range", self.lesson_range),
                readings_dir=self.reading_dir,
                syllabus_path=self.syllabus_path,
                llm=self.llm,
                project_dir=self.project_dir,
                course_name=kwargs.get("course_name", "Political Science"),
                output_dir=concept_output_dir,  # Allow for custom output_dir
                verbose=kwargs.get("verbose", False),  # Allow additional kwargs like verbosity
                recursive=True  # go down one directory to find the lesson
            )

        elif module_name in ["QuizMaker", "quizmaker"]:
            # all outputs (quizzes generated, quiz analysis, etc) will be placed in the lesson for which they were run
            quiz_output_dir = interim_output_dir / f"QuizMaker/L{self.lesson_no}"
            quiz_output_dir.mkdir(parents=True, exist_ok=True)
            return QuizMaker(
                lesson_range=kwargs.get("lesson_range", self.lesson_range),
                llm=self.llm,
                syllabus_path=self.syllabus_path,
                reading_dir=self.reading_dir,
                output_dir=quiz_output_dir,
                course_name=kwargs.get("course_name", "Political Science"),
                prior_quiz_path=self.project_dir / "data/quizzes",
                verbose=kwargs.get("verbose", False),
            )  # Single lesson by default
        else:
            raise ValueError(f"Module {module_name} not recognized.")


if __name__ == "__main__":
    import os

    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from pyprojroot.here import here
    wd = here()

    OPENAI_KEY = os.getenv('openai_key')
    OPENAI_ORG = os.getenv('openai_org')

    lesson_no = 21

    # Path definitions
    readingDir = Path(os.getenv('readingsDir'))
    slideDir = Path(os.getenv('slideDir'))
    syllabus_path = Path(os.getenv('syllabus_path'))

    input_dir = readingDir / f'L{lesson_no}/'
    beamer_example = slideDir / f'L{lesson_no-1}.tex'
    beamer_output = slideDir / f'L{lesson_no}.tex'

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_KEY,
        organization=OPENAI_ORG,
    )
    # llm = Ollama(
    #     model="llama3.1",
    #     temperature=0.2
    # )

    # Initialize the factory
    factory = ClassFactory(lesson_no=lesson_no,
                           syllabus_path=syllabus_path,
                           reading_dir=readingDir,
                           slide_dir=slideDir,
                           llm=llm,
                           project_dir=wd,
                           lesson_range=range(17, 21),
                           course_name="American Government")
    # %%
    # build slides

    specific_guidance = """
    The lesson should be structured in a way that discusses big picture ideas about political campaigns.
    The slides will, at a minimum, cover the following (using the readings as a reference):
               - How campaigns operate
               - How our party system came to be
               - Individual vote choice
               - The types and determinants of today's polarization​
    """

    beamerbot = factory.create_module("BeamerBot", verbose=False)
    slides = beamerbot.generate_slides()           # specific guidance might make the results more generic
    # beamerbot.save_slides(slides)

    # # build concept map
    # builder = factory.create_module("ConceptWeb",
    #                                 course_name = "American Government")

    # builder.build_concept_map()

    # quizmaker = factory.create_module("QuizMaker")

    # quiz = quizmaker.make_a_quiz()
    # quizmaker.launch_interactive_quiz(quiz_data=quiz, sample_size=7)
