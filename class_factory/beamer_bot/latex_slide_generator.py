"""
**LatexSlideGenerator Module**
-------------------------------

The `LatexSlideGenerator` module provides LaTeX Beamer slide generation based on lesson objectives,
readings, and prior lesson presentations. This is the concrete implementation of the BaseSlideGenerator
for LaTeX/Beamer output format.

Key Functionalities
~~~~~~~~~~~~~~~~~~~

1. **Automated LaTeX Beamer Slide Generation**:
   - Generates a LaTeX Beamer presentation for each lesson, incorporating:
     - A title page with consistent author and institution information
     - "Where We Came From" and "Where We Are Going" slides
     - Lesson objectives with highlighted action verbs
     - Discussion questions and in-class exercises
     - Summary slides with key takeaways

2. **Previous Lesson Integration**:
   - Retrieves and references prior lesson presentations to maintain consistent formatting and flow
   - Preserves author and institution information across presentations

3. **LaTeX-Specific Validation**:
   - Validates generated LaTeX for correct formatting and content quality
   - Provides multiple retry attempts if validation fails
   - Compiles LaTeX to verify syntax correctness

Usage
~~~~~~~

1. **Initialize LatexSlideGenerator**:
   ```python
   generator = LatexSlideGenerator(
       lesson_no=10,
       llm=llm,
       course_name="Political Science",
       lesson_loader=lesson_loader,
       output_dir=output_dir
   )
   ```

2. **Generate Slides**:
   ```python
   # Optional specific guidance
   guidance = "Focus on comparing democratic and authoritarian systems"
   slides = generator.generate_slides(specific_guidance=guidance)
   ```

3. **Save the Slides**:
   ```python
   generator.save_slides(slides)
   ```
"""
# %%
import logging
from pathlib import Path
from typing import Any, Dict, Union

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate)

from class_factory.beamer_bot.base_slide_generator import BaseSlideGenerator
from class_factory.beamer_bot.latex_prompts import (latex_human_prompt,
                                                    latex_system_prompt)
from class_factory.beamer_bot.latex_slide_preamble import preamble
from class_factory.beamer_bot.latex_slides import LatexSlides
from class_factory.beamer_bot.latex_utils import (comment_out_includegraphics,
                                                  validate_latex)
from class_factory.utils.load_documents import LessonLoader


class LatexSlideGenerator(BaseSlideGenerator):
    """
    A class to generate LaTeX Beamer slides for a specified lesson using a language model (LLM).

    LatexSlideGenerator automates the slide generation process, creating structured presentations
    based on lesson readings, objectives, and content from prior presentations when available.
    Each slide is crafted following a consistent format, and the generated LaTeX is validated
    for correctness.

    Attributes:
        lesson_no (int): Lesson number for which to generate slides.
        llm: Language model instance for generating slides.
        course_name (str): Name of the course for slide context.
        lesson_loader (LessonLoader): Loader for accessing lesson readings and objectives.
        output_dir (Path): Directory to save the generated Beamer slides.
        slide_dir (Path): Directory containing existing Beamer slides.
        llm_response (str): Stores the generated LaTeX response from the LLM.
        prompt (ChatPromptTemplate): Generated prompt for the LLM.
        chain: LLM chain with structured output for LaTeX slides.
        beamer_output (Path): Output path for the generated .tex file.

    Methods:
        generate_slides(specific_guidance: str = None, latex_compiler: str = "pdflatex") -> str:
            Generates Beamer slides as LaTeX code for the specified lesson.

        save_slides(latex_content: str) -> None:
            Saves the generated LaTeX content to a .tex file.

    Internal Methods:
        _load_prior_lesson() -> str:
            Loads the LaTeX content of a prior lesson's Beamer presentation as a string.

        _generate_prompt() -> ChatPromptTemplate:
            Constructs the LLM prompt using lesson objectives, readings, and prior lesson content.

        _validate_llm_response(generated_slides: str, objectives: str, readings: str,
                               last_presentation: str, prompt_specific_guidance: str = "",
                               additional_guidance: str = "") -> Dict[str, Any]:
            Validates the generated LaTeX for quality and accuracy.
    """

    def __init__(self, lesson_no: int, llm, course_name: str, lesson_loader: LessonLoader,
                 output_dir: Union[Path, str] = None, verbose: bool = False,
                 slide_dir: Union[Path, str] = None, lesson_objectives: dict = None):
        """
        Initialize the LaTeX slide generator.

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
            llm=llm,
            course_name=course_name,
            lesson_loader=lesson_loader,
            output_dir=output_dir,
            verbose=verbose,
            slide_dir=slide_dir,
            lesson_objectives=lesson_objectives
        )

        # Generate the prompt
        self.prompt = self._generate_prompt()

        # Use LLM with structured output for LaTeX slides
        self.chain = self.prompt | self.llm.with_structured_output(LatexSlides)

        # Set output file path
        self.beamer_output = self.output_dir / f'L{self.lesson_no}.tex'

    def _load_prior_lesson(self) -> str:
        """
        Load the previous lesson's Beamer presentation as a string.

        Returns:
            str: Content of the prior lesson's LaTeX presentation
        """
        beamer_example = self.lesson_loader.find_prior_beamer_presentation(self.lesson_no)
        return self.lesson_loader.load_beamer_presentation(beamer_example)

    def _generate_prompt(self, human_prompt: str = None) -> ChatPromptTemplate:
        """
        Generates a detailed prompt for the LLM to guide LaTeX Beamer slide creation.

        Args:
            human_prompt (str, optional): Override default human prompt template

        Returns:
            ChatPromptTemplate: The constructed prompt for the LLM.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        latex_system_prompt.format(
                            course_name=self.course_name)
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    latex_human_prompt if not human_prompt else human_prompt),
            ]
        )

        return prompt

    def generate_slides(self, specific_guidance: str = None, lesson_objectives: dict = None,
                        latex_compiler: str = "pdflatex") -> str:
        """
        Generate LaTeX Beamer slides for the lesson using the language model.

        Args:
            specific_guidance (str, optional): Custom instructions for slide content and structure
            lesson_objectives (dict, optional): Override default objectives with custom ones
                Format: {lesson_number: "objective text"}
            latex_compiler (str, optional): LaTeX compiler to use for validation. Defaults to "pdflatex"

        Returns:
            str: Complete LaTeX content for the presentation, including preamble

        Raises:
            ValueError: If validation fails after maximum retry attempts
            FileNotFoundError: If required prior lesson files cannot be located

        Note:
            The method includes multiple validation steps:
            1. Content quality validation through LLM
            2. LaTeX syntax validation using specified compiler
            3. Up to 3 retry attempts if validation fails
        """
        # Load objectives (last, current, next), readings, and previous lesson slides
        self.user_objectives = self.set_user_objectives(lesson_objectives, range(self.lesson_no, self.lesson_no+1)) if lesson_objectives else {}
        objectives_text = "\n\n".join([self._get_lesson_objectives(lesson)
                                      for lesson in range(self.lesson_no - 1, self.lesson_no + 2)])
        combined_readings_text = self.readings

        if self.lesson_loader.slide_dir:
            prior_lesson_tex = self._load_prior_lesson()
        else:
            prior_lesson_tex = "Not Provided"
            self.logger.warning(
                "No slide_dir provided. Prior slides will not be referenced during generation. "
                "If this is unintentional, please check LessonLoader configuration for slide_dir."
            )

        self.logger.info(f"{self.prompt=}")
        # Generate Beamer slides via the chain
        additional_guidance = ""
        retries, MAX_RETRIES = 0, 3
        valid = False

        while not valid and retries < MAX_RETRIES:
            # LLM returns LatexSlides object
            slides_data = self.chain.invoke({
                "objectives": objectives_text,
                "information": combined_readings_text,
                "last_presentation": prior_lesson_tex,
                "lesson_no": self.lesson_no,
                'specific_guidance': specific_guidance if specific_guidance else "Not provided.",
                "additional_guidance": additional_guidance
            })

            # Parse LLM output into LaTeX
            latex_body = slides_data.to_latex()
            full_latex = preamble + "\n\n" + comment_out_includegraphics(latex_body)
            self.llm_response = full_latex

            # Validate the structured output and LaTeX
            val_response = self._validate_llm_response(
                generated_slides=slides_data,
                objectives=objectives_text,
                readings=combined_readings_text,
                last_presentation=prior_lesson_tex,
                prompt_specific_guidance=specific_guidance if specific_guidance else "Not provided.",
                task_schema=LatexSlides.model_json_schema()
            )
            self.validator.validation_result = val_response
            self.validator.logger.info(f"Validation output: {val_response}")

            # Use both status and overall_score for validation
            if int(val_response.get('status', 0)) != 1:
                retries += 1
                additional_guidance = val_response.get("additional_guidance", "")
                self.validator.logger.warning(
                    f"Response validation failed on attempt {retries}. "
                    f"Guidance for improvement: {additional_guidance}"
                )
                continue  # Retry LLM generation

            # Validate the generated LaTeX code
            is_valid_latex = False  # Reset each iteration
            try:
                is_valid_latex = validate_latex(full_latex, latex_compiler=latex_compiler)
            except Exception as e:
                self.logger.error(f"LaTeX validation encountered an error: {e}")

            if is_valid_latex:
                valid = True
                return full_latex
            else:
                retries += 1  # Increment retries only if validation fails
                self.logger.warning("\nLaTeX code is invalid. Attempting a second model run. "
                                    "If the error persists, please review the LLM output for potential causes. "
                                    "You can inspect the model output via the 'llm_response' object (LatexSlideGenerator.llm_response). "
                                    "\n\nNote: Compilation issues may stem from syntax errors in the example LaTeX code provided to the model."
                                    )

        # Handle validation failure after max retries
        if not valid:
            raise ValueError("Validation failed after max retries. Ensure correct prompt and input data. Consider trying a different LLM.")

    def _validate_llm_response(self, generated_slides, objectives: str, readings: str, last_presentation: str,
                               prompt_specific_guidance: str = "", additional_guidance: str = "", task_schema=None) -> Dict[str, Any]:
        """
        Validates the generated LaTeX slides for content quality and formatting accuracy.

        Args:
            generated_slides (LatexSlides): LaTeX slides generated by the LLM
            objectives (str): Formatted string of lesson objectives for validation
            readings (str): Formatted string of lesson readings for content verification
            last_presentation (str): Content from prior lesson's presentation for format consistency
            prompt_specific_guidance (str, optional): Custom guidance provided during generation
            additional_guidance (str, optional): Supplementary guidance for validation refinement
            task_schema: JSON schema for the LatexSlides model

        Returns:
            Dict[str, Any]: Validation results containing:
                - status (int): 1 for pass, 0 for fail
                - evaluation_score (float): Quality score (0-10)
                - additional_guidance (str): Suggestions for improvement if validation fails
                - rationale (str): Explanation of the validation result

        Note:
            The validation process uses the same prompt template as slide generation to ensure
            consistency between requirements and validation criteria.
        """
        # Validate slide quality and accuracy
        response_str = str(generated_slides)
        validation_prompt = self.prompt.format(
            objectives=objectives,
            information=readings,
            last_presentation=last_presentation,
            lesson_no=self.lesson_no,
            prior_lesson=self.prior_lesson,
            additional_guidance=additional_guidance,
            specific_guidance=prompt_specific_guidance
        )

        val_response = self.validator.validate(
            task_description=validation_prompt,
            generated_response=generated_slides,
            task_schema=task_schema,
            specific_guidance=prompt_specific_guidance + ("\n" + additional_guidance if additional_guidance else "")
        )
        return val_response

    def save_slides(self, latex_content: str, output_dir: Union[Path, str] = None) -> None:
        """
        Save the generated LaTeX content to a .tex file.

        Args:
            latex_content (str): The LaTeX content to save.
            output_dir (Union[Path, str], optional): Override default output directory
        """
        if output_dir:
            beamer_file = Path(output_dir) / f'L{self.lesson_no}.tex'
            with open(beamer_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
        else:
            with open(self.beamer_output, 'w', encoding='utf-8') as f:
                f.write(latex_content)
        self.logger.info(f"Slides saved to {self.beamer_output if not output_dir else beamer_file}")


# %%
if __name__ == "__main__":
    import os

    import yaml
    from dotenv import load_dotenv
    from pyprojroot.here import here

    from class_factory.utils.tools import get_llm, reset_loggers

    wd = here()
    load_dotenv()

    user_home = Path.home()
    reset_loggers(log_level=logging.INFO)

    # Path definitions
    with open("class_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # class_config = config['PS491']
    class_config = config['PS300']

    slide_dir = user_home / class_config['slideDir']
    syllabus_path = user_home / class_config['syllabus_path']
    readingsDir = user_home / class_config['reading_dir']
    is_tabular_syllabus = class_config['is_tabular_syllabus']

    # Initialize LLM using get_llm helper
    # Options: "openai", "anthropic", "gemini", "ollama"
    llm = get_llm("gemini")

    lsn = 3

    specific_guidance = """
    The objectives slide should include an objective titled "have tons of fun"
    """

    loader = LessonLoader(syllabus_path=syllabus_path,
                          reading_dir=readingsDir,
                          slide_dir=slide_dir,
                          tabular_syllabus=is_tabular_syllabus
                          )

    # Initialize the BeamerBot (defaults to LaTeX)
    beamer_bot = LatexSlideGenerator(
        lesson_no=2,
        lesson_loader=loader,
        llm=llm,
        course_name="Political Science Research Methods",
        verbose=True
    )

    # Generate slides for Lesson 11
    slides = beamer_bot.generate_slides()

    print(slides)
    # Save the generated LaTeX slides
    # beamer_bot.save_slides(slides)


# %%
