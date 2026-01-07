"""
**BeamerBot Module**
--------------------

The `BeamerBot` module provides a framework for generating structured presentation slides in multiple formats
(LaTeX Beamer, PowerPoint, etc.) based on lesson objectives, readings, and prior lesson presentations.
By using a language model (LLM), `BeamerBot` automates the process of slide creation, ensuring a consistent
slide structure while allowing for custom guidance and validation.

BeamerBot acts as a factory that dispatches to the appropriate slide generator based on the requested output format.

Key Functionalities
~~~~~~~~~~~~~~~~~~~

1. **Multi-Format Slide Generation**:
   - Supports LaTeX Beamer presentations (.tex)
   - Supports PowerPoint presentations (.pptx) [coming soon]
   - Easily extensible to other formats

2. **Automated Content Generation**:
   - Generates presentations incorporating:
     - A title page with consistent author and institution information
     - "Where We Came From" and "Where We Are Going" slides
     - Lesson objectives with highlighted action verbs
     - Discussion questions and in-class exercises
     - Summary slides with key takeaways

3. **Previous Lesson Integration**:
   - Retrieves and references prior lesson presentations to maintain consistent formatting and flow
   - Preserves author and institution information across presentations

4. **Format-Specific Processing**:
   - Each format has its own generator with specialized prompts and validation
   - LaTeX: Validates LaTeX syntax and compiles to PDF
   - PowerPoint: Uses python-pptx for slide creation [coming soon]

Dependencies
~~~~~~~~~~~~~

This module requires:

- `pathlib`: For file path management
- Format-specific generators:
  - `LatexSlideGenerator`: For LaTeX Beamer output
  - `PptxSlideGenerator`: For PowerPoint output [coming soon]
- Custom utility modules for document loading and validation

Usage
~~~~~~~

1. **Initialize BeamerBot** (defaults to LaTeX):
   ```python
   # LaTeX output (default)
   beamer_bot = BeamerBot(
       lesson_no=10,
       llm=llm,
       course_name="Political Science",
       lesson_loader=lesson_loader,
       output_dir=output_dir
   )

   # Or explicitly specify format
   beamer_bot = BeamerBot(
       lesson_no=10,
       llm=llm,
       output_format="latex",  # or "pptx"
       course_name="Political Science",
       lesson_loader=lesson_loader,
       output_dir=output_dir
   )
   ```

2. **Generate Slides**:
   ```python
   # Optional specific guidance
   guidance = "Focus on comparing democratic and authoritarian systems"
   slides = beamer_bot.generate_slides(specific_guidance=guidance)
   ```

3. **Save the Slides**:
   ```python
   beamer_bot.save_slides(slides)
   ```

Architecture
~~~~~~~~~~~~

BeamerBot uses the Factory pattern:
- `BeamerBot`: Factory class that creates the appropriate generator
- `BaseSlideGenerator`: Abstract base class with common functionality
- `LatexSlideGenerator`: Concrete implementation for LaTeX Beamer
- `PptxSlideGenerator`: Concrete implementation for PowerPoint [coming soon]
"""
# %%
import logging
from pathlib import Path
from typing import Union

from class_factory.beamer_bot.latex_slide_generator import LatexSlideGenerator
from class_factory.beamer_bot.pptx_slide_generator import PptxSlideGenerator
from class_factory.utils.load_documents import LessonLoader


class BeamerBot:
    """
    Factory class for creating slide generators in various output formats.

    BeamerBot maintains backward compatibility with the original API while providing
    flexibility to generate slides in multiple formats (LaTeX, PowerPoint, etc.).

    The class uses Python's __new__ method to return the appropriate generator instance
    based on the requested output_format parameter.

    Supported Formats:
        - "latex": LaTeX Beamer presentations (.tex files)
        - "pptx": PowerPoint presentations (.pptx files) [coming soon]

    Args:
        output_format (str): The desired output format. Options: "latex", "pptx".
                            Defaults to "latex" for backward compatibility.
        lesson_no (int): Lesson number for which to generate slides.
        llm: Language model instance for generating slides.
        course_name (str): Name of the course for slide context.
        lesson_loader (LessonLoader): Loader for accessing lesson readings and objectives.
        output_dir (Union[Path, str], optional): Directory to save the generated slides.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
        slide_dir (Union[Path, str], optional): Directory containing existing slides.
        lesson_objectives (dict, optional): User-provided lesson objectives.
        **kwargs: Additional format-specific arguments passed to the generator.

    Returns:
        An instance of the appropriate slide generator (LatexSlideGenerator or PptxSlideGenerator).

    Raises:
        ValueError: If an unsupported output_format is specified.

    Examples:
        >>> # Create LaTeX generator (default)
        >>> bot = BeamerBot(lesson_no=10, llm=llm, lesson_loader=loader, course_name="Politics")
        >>> slides = bot.generate_slides()
        >>>
        >>> # Create PowerPoint generator
        >>> bot = BeamerBot(output_format="pptx", lesson_no=10, llm=llm, lesson_loader=loader, course_name="Politics")
        >>> slides = bot.generate_slides()
    """

    def __new__(
        cls,
        output_format: str = "latex",
        lesson_no: int = None,
        llm=None,
        course_name: str = None,
        lesson_loader: LessonLoader = None,
        output_dir: Union[Path, str] = None,
        verbose: bool = False,
        slide_dir: Union[Path, str] = None,
        lesson_objectives: dict = None,
        **kwargs
    ):
        """
        Create and return the appropriate slide generator based on output_format.

        This method intercepts object creation and returns an instance of the
        appropriate generator class instead of a BeamerBot instance.
        """
        # Validate required parameters
        if lesson_no is None:
            raise ValueError("lesson_no is required")
        if llm is None:
            raise ValueError("llm is required")
        if course_name is None:
            raise ValueError("course_name is required")
        if lesson_loader is None:
            raise ValueError("lesson_loader is required")

        # Normalize output format
        output_format = output_format.lower().strip()

        # Common parameters for all generators
        generator_params = {
            "lesson_no": lesson_no,
            "llm": llm,
            "course_name": course_name,
            "lesson_loader": lesson_loader,
            "output_dir": output_dir,
            "verbose": verbose,
            "slide_dir": slide_dir,
            "lesson_objectives": lesson_objectives,
        }

        # Add any additional kwargs
        generator_params.update(kwargs)

        # Dispatch to the appropriate generator
        if output_format == "latex":
            return LatexSlideGenerator(**generator_params)
        elif output_format == "pptx" or output_format == "powerpoint":
            return PptxSlideGenerator(**generator_params)
        else:
            raise ValueError(
                f"Unsupported output format: '{output_format}'. "
                f"Supported formats: 'latex', 'pptx'"
            )


# %%
if __name__ == "__main__":
    import os

    import yaml
    from dotenv import load_dotenv
    from langchain_community.llms import Ollama
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
    from pyprojroot.here import here

    from class_factory.utils.tools import reset_loggers

    wd = here()
    load_dotenv()

    user_home = Path.home()
    reset_loggers(log_level=logging.INFO)

    OPENAI_KEY = os.getenv('openai_key')
    OPENAI_ORG = os.getenv('openai_org')
    GEMINI_KEY = os.getenv('gemini_api_key')

    # Path definitions
    with open("class_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # class_config = config['PS491']
    class_config = config['PS300']

    slide_dir = user_home / class_config['slideDir']
    syllabus_path = user_home / class_config['syllabus_path']
    readingsDir = user_home / class_config['reading_dir']
    is_tabular_syllabus = class_config['is_tabular_syllabus']

    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0.4,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     api_key=OPENAI_KEY,
    #     organization=OPENAI_ORG,
    # )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GEMINI_KEY
    )

    lsn = 3

    # llm = Ollama(
    #     model="llama3.1",
    #     temperature=0.2
    # )

    specific_guidance = """
    The objectives slide should include an objective titled "have tons of fun"
    """

    loader = LessonLoader(syllabus_path=syllabus_path,
                          reading_dir=readingsDir,
                          slide_dir=slide_dir,
                          tabular_syllabus=is_tabular_syllabus
                          )

    # Initialize the BeamerBot (defaults to LaTeX)
    beamer_bot = BeamerBot(
        output_format="pptx",
        lesson_no=2,
        lesson_loader=loader,
        llm=llm,
        course_name="Political Science Research Methods",
        verbose=True
    )

    # Generate slides for Lesson 11
    slides = beamer_bot.generate_slides(lesson_objectives={"11": "Learn Feaver's principal-agent theory"},)

    print(slides)
    # Save the generated  slides
    beamer_bot.save_slides(slides)

# %%
