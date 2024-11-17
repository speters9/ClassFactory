"""
**ClassFactory Module**
-------

The `ClassFactory` module is designed to provide a unified interface for integrating and managing various educational modules powered by AI. These modules include:

- **BeamerBot**: Automates the generation of LaTeX Beamer slides for lessons, based on lesson readings and objectives.
- **ConceptWeb**: Generates a concept map of key ideas and their relationships across lessons.
- **QuizMaker**: Creates quizzes based on lesson materials, with options for interactive quizzes, results analysis, and similarity checks with prior quizzes.

Key Functionalities
~~~~~~~

1. **Module Instantiation**:
   - `ClassFactory` includes a `create_module` method to dynamically create and return instances of `BeamerBot`, `ConceptWeb`, or `QuizMaker`.
   - Each module instance inherits the current lesson context, paths, and configurations defined during `ClassFactory` initialization.

2. **Centralized Output Management**:
   - By default, all module outputs are saved in an organized directory structure within the `ClassFactoryOutput` folder.
   - Each module’s output (e.g., Beamer slides, concept maps, quizzes) is saved in its respective directory, organized by lesson number for easy access.

3. **Integrated LLM**:
   - A language model (e.g., GPT-4, LLaMA) is passed to each module to support automated content generation based on lesson readings and objectives.

Usage
~~~~~~~

1. **Initialization**:
   - Initialize the `ClassFactory` with the lesson number, paths to the syllabus, readings, slides, and an LLM instance. Optionally, specify the project and output directories.

2. **Creating Modules**:
   - Use the `create_module` method to instantiate a specific module (`BeamerBot`, `ConceptWeb`, or `QuizMaker`) and access its methods for generating slides, concept maps, or quizzes.

3. **Customization**:
   - Additional configurations can be provided as keyword arguments for each module (e.g., custom output directory, lesson range).

Dependencies
~~~~~~~

- `BeamerBot`: Manages LaTeX slide creation.
- `ConceptWeb`: Builds concept maps from lesson objectives and readings.
- `QuizMaker`: Generates and analyzes quizzes.
- Custom utilities for loading documents, resetting loggers, and managing file paths.

Example
~~~~~~~

.. code-block:: python

    from class_factory import ClassFactory
    from langchain_openai import ChatOpenAI
    from pathlib import Path

    # Initialize paths
    syllabus_path = Path("/path/to/syllabus.docx")
    reading_dir = Path("/path/to/readings")
    slide_dir = Path("/path/to/slides")
    output_dir = Path("/path/to/output")
    llm = ChatOpenAI(api_key="your_api_key")

    # Instantiate ClassFactory
    factory = ClassFactory(
        lesson_no=10,
        syllabus_path=syllabus_path,
        reading_dir=reading_dir,
        llm=llm,
        project_dir=output_dir
    )

    # Generate slides with BeamerBot
    beamerbot = factory.create_module("BeamerBot")
    slides = beamerbot.generate_slides()
    beamerbot.save_slides(slides)

    # Build a concept map with ConceptWeb
    conceptweb = factory.create_module("ConceptWeb")
    conceptweb.build_concept_map()

    # Create a quiz with QuizMaker
    quizmaker = factory.create_module("QuizMaker")
    quiz = quizmaker.make_a_quiz()
    quizmaker.launch_interactive_quiz(quiz_data=quiz, sample_size=5)


"""

from pathlib import Path
from typing import Optional, Union

from pyprojroot.here import here

from class_factory.utils.load_documents import LessonLoader
# from class_factory.beamer_bot.BeamerBot import BeamerBot
# from class_factory.concept_web.ConceptWeb import ConceptMapBuilder
# from class_factory.quiz_maker.QuizMaker import QuizMaker
from class_factory.utils.tools import reset_loggers

reset_loggers()


class ClassFactory:
    """
    A factory class responsible for creating and managing instances of various educational modules.

    `ClassFactory` provides a standardized interface for initializing educational modules designed for generating
    lesson-specific materials, such as slides, concept maps, and quizzes. Modules are dynamically created based on the
    specified module name, with configurations for content generation provided by the user.

    Modules available for creation include:
    - **BeamerBot**: Automated LaTeX Beamer slide generation.
    - **ConceptWeb**: Concept map generation based on lesson objectives and readings.
    - **QuizMaker**: Quiz creation, hosting, and analysis.

    Attributes:
        lesson_no (int): The lesson number for which the module instance is created.
        syllabus_path (Path): Path to the syllabus file.
        reading_dir (Path): Path to the directory containing lesson readings.
        slide_dir (Path): Path to the directory containing lesson slides.
        llm: Language model instance used for content generation in modules.
        project_dir (Path): Base project directory.
        output_dir (Path): Directory where outputs from modules are saved.
        lesson_range (range): Range of lessons covered by the factory instance.
        course_name (str): Name of the course for which content is generated.
        lesson_loader (LessonLoader): Instance of `LessonLoader` for loading lesson-related data and objectives.

    By default, all module outputs are saved in a structured directory under "ClassFactoryOutput" within the project directory.
    """

    def __init__(self, lesson_no: int, syllabus_path: Union[str, Path], reading_dir: Union[str, Path],
                 llm, project_dir: Optional[Union[str, Path]] = None, output_dir: Optional[Union[str, Path]] = None,
                 slide_dir: Optional[Union[str, Path]] = None, lesson_range: Optional[range] = None,
                 course_name: str = "Political Science", verbose: bool = True, **kwargs):
        """
        Initialize ClassFactory with paths, configurations, and an optional lesson range for multi-lesson processing.

        Args:
            lesson_no (int): The lesson number for which to create modules.
            syllabus_path (Union[str, Path]): Path to the syllabus file.
            reading_dir (Union[str, Path]): Path to the directory containing lesson readings.
            llm: Language model used for generating content in the modules.
            project_dir (Optional[Union[str, Path]]): Base project directory. Defaults to current directory.
            output_dir (Optional[Union[str, Path]]): Directory where output files are saved; defaults to 'ClassFactoryOutput'.
            slide_dir (Optional[Union[str, Path]]): Directory containing slide files for the lesson.
            lesson_range (Optional[range]): Range of lessons to cover. Defaults to a single lesson.
            course_name (str): Name of the course for context in content generation. Defaults to "Political Science".
            verbose (bool): If True, enables verbose logging in `LessonLoader`.
            **kwargs: Additional configurations for modules.
        """
        self.lesson_no = lesson_no
        self.lesson_range = lesson_range if lesson_range else range(lesson_no, lesson_no + 1)  # Default to a single lesson
        self.course_name = course_name
        self.llm = llm
        self.output_dir = Path(output_dir) if output_dir else here() / "ClassFactoryOutput"
        self.lesson_loader = LessonLoader(
            syllabus_path=syllabus_path,
            reading_dir=reading_dir,
            slide_dir=slide_dir if slide_dir else None,
            project_dir=Path(project_dir) if project_dir else here(),
            verbose=verbose
        )

    def create_module(self, module_name: str, **kwargs):
        """
        Create a specific module instance based on the provided module name.

        Args:
            module_name (str): The name of the module to create. Options are 'BeamerBot', 'ConceptWeb', or 'QuizMaker'.
            **kwargs: Additional keyword arguments for the specific module, including custom configurations for verbosity, output directory, or lesson range.

        Returns:
            Union[BeamerBot, ConceptMapBuilder, QuizMaker]: The created module instance based on the provided name.

        Raises:
            ValueError: If an invalid module name is provided.

        By default, each module output will be saved in its respective directory in the ClassFactoryOutput/ directory,
        in a directory named after the designated lesson number.
        """
        interim_output_dir = kwargs.get("output_dir", self.output_dir)
        if module_name in ["BeamerBot", "beamerbot"]:
            try:
                from class_factory.beamer_bot.BeamerBot import BeamerBot
            except ImportError as e:
                raise ImportError("Failed to import BeamerBot. Please check that the module is available.") from e

            beamer_output_dir = interim_output_dir / f"BeamerBot/L{self.lesson_no}"
            beamer_output_dir.mkdir(parents=True, exist_ok=True)
            # BeamerBot should still use a single lesson (lesson_no)
            return BeamerBot(
                lesson_no=self.lesson_no,
                llm=self.llm,
                lesson_loader=self.lesson_loader,
                output_dir=beamer_output_dir,
                verbose=kwargs.get("verbose", False),
                course_name=kwargs.get("course_name", self.course_name),
                slide_dir=kwargs.get("slide_dir", self.lesson_loader.slide_dir)
            )
        elif module_name in ["ConceptWeb", "conceptweb"]:
            try:
                from class_factory.concept_web.ConceptWeb import \
                    ConceptMapBuilder
            except ImportError as e:
                raise ImportError("Failed to import ConceptWeb. Please check that the module is available.") from e

            concept_output_dir = interim_output_dir / f"ConceptWeb/L{self.lesson_no}"
            concept_output_dir.mkdir(parents=True, exist_ok=True)
            # ConceptMapBuilder uses the lesson range
            return ConceptMapBuilder(
                lesson_no=self.lesson_no,
                lesson_range=kwargs.get("lesson_range", self.lesson_range),
                lesson_loader=self.lesson_loader,
                llm=self.llm,
                course_name=kwargs.get("course_name", self.course_name),
                output_dir=concept_output_dir,  # Allow for custom output_dir
                verbose=kwargs.get("verbose", False),  # Allow additional kwargs like verbosity
            )

        elif module_name in ["QuizMaker", "quizmaker"]:
            try:
                from class_factory.quiz_maker.QuizMaker import QuizMaker
            except ImportError as e:
                raise ImportError("Failed to import QuizMaker. Please check that the module is available.") from e

            # all outputs (quizzes generated, quiz analysis, etc) will be placed in the lesson for which they were run
            quiz_output_dir = interim_output_dir / f"QuizMaker/L{self.lesson_no}"
            quiz_output_dir.mkdir(parents=True, exist_ok=True)
            return QuizMaker(
                lesson_range=kwargs.get("lesson_range", self.lesson_range),
                lesson_no=kwargs.get("lesson_no", self.lesson_range),
                lesson_loader=self.lesson_loader,
                llm=self.llm,
                output_dir=quiz_output_dir,
                course_name=kwargs.get("course_name", self.course_name),
                prior_quiz_path=self.lesson_loader.project_dir / "data/quizzes",
                verbose=kwargs.get("verbose", False),
            )  # Single lesson by default
        else:
            raise ValueError(f"Module {module_name} not recognized.")


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from pyprojroot.here import here

    user_home = Path.home()
    load_dotenv()
    wd = here()

    OPENAI_KEY = os.getenv('openai_key')
    OPENAI_ORG = os.getenv('openai_org')

    lesson_no = 21

    # Path definitions
    readingDir = user_home / os.getenv('readingsDir')
    slideDir = user_home / os.getenv('slideDir')
    syllabus_path = user_home / os.getenv('syllabus_path')

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

    # beamerbot = factory.create_module("BeamerBot", slide_dir=slideDir, verbose=True)
    # slides = beamerbot.generate_slides()           # specific guidance might make the results more generic
    # beamerbot.save_slides(slides)

    # # build concept map
    builder = factory.create_module("ConceptWeb")

    builder.build_concept_map(directed=True)

    # quizmaker = factory.create_module("QuizMaker")

    # quiz = quizmaker.make_a_quiz()
    # quizmaker.launch_interactive_quiz(quiz_data=quiz, sample_size=7)
