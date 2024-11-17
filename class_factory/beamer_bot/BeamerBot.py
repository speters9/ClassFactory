"""
**BeamerBot**

This module defines the `BeamerBot` class, which automates the creation of LaTeX Beamer slides for lessons, leveraging a language model (LLM) to generate content based on lesson readings, objectives, and past presentations.

The module includes the following key functionalities:

- **Automated Slide Generation**:

    BeamerBot generates structured LaTeX Beamer slides for each lesson, which include placeholders for specific sections such as:
    - A current event slide after the title page.
    - A slide outlining the lesson objectives, with actions in bold (e.g., "Understand the role of government").
    - A discussion question slide relevant to the lesson content.
    - Primary takeaway slides summarizing the main points of the lesson.

- **Previous Lesson Retrieval**:

    Automatically retrieves the LaTeX file of the previous lesson as well as current lesson objectives to ensure continuity in slide structure and formatting. If the previous lesson is unavailable, it attempts to retrieve one from earlier lessons.

- **Customizable Prompts**:

    The language model is prompted using structured lesson objectives and readings, allowing for custom guidance on what content to generate for each slide.

- **Validation and Output**:

    The generated LaTeX is validated to ensure it follows correct formatting before saving to a `.tex` file.

Dependencies:

- `langchain_core`: For integrating the language model and managing prompt templates.
- Custom utilities for reading objectives, validating file paths, loading documents, and logging.

Usage:

1. **Initialize BeamerBot**:

    Create an instance of `BeamerBot` with the lesson number, syllabus path, reading directory, and slide directory. Optionally, you can provide a custom language model.

2. **Generate Slides**:

    Call the `generate_slides()` method to produce the LaTeX code for the lesson slides.

3. **Save the Slides**:

    Use the `save_slides()` method to save the generated LaTeX content to a file.
"""


import logging
import os
from pathlib import Path
from typing import Any, Dict, Union

# env setup
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pyprojroot.here import here

# base libraries
from class_factory.beamer_bot.slide_preamble import preamble
from class_factory.utils.base_model import BaseModel
from class_factory.utils.llm_validator import Validator
from class_factory.utils.load_documents import LessonLoader
from class_factory.utils.response_parsers import ValidatorResponse
from class_factory.utils.slide_pipeline_utils import (
    clean_latex_content, comment_out_includegraphics, validate_latex)

# %%


class BeamerBot(BaseModel):
    """
    A class to generate LaTeX Beamer slides for a given lesson using a language model (LLM).

    Attributes:
        lesson_no (int): Lesson number for generating slides.
        syllabus_path (Path): Path to the syllabus file.
        reading_dir (Path): Directory where lesson readings are stored.
        slide_dir (Path): Directory where Beamer slides are stored.
        llm: Language model for generating slides.
        output_dir (Path): Directory to save the output Beamer slides.
        prompt (str): Generated prompt for the LLM.

    Methods:
        generate_slides(specific_guidance: str = None, latex_compiler: str = "pdflatex") -> str:
            Generate the Beamer slides for the specified lesson using the LLM.

        save_slides(latex_content: str) -> None:
            Save the generated LaTeX content to a .tex file.

    Internal Methods:
        _load_readings() -> str:
            Load lesson readings from the specified directory.

        _find_prior_lesson(lesson_no: int, max_attempts: int = 3) -> Path:
            Find the most recent prior lesson's Beamer file to use as a slide template.

        _load_prior_lesson() -> str:
            Load and return the prior lesson's Beamer presentation content as a string.

        _generate_prompt() -> str:
            Generate the LLM prompt using lesson objectives, readings, and prior lesson content.

        _validate_llm_response(generated_slides: str, objectives: str, readings: str, last_presentation: str, prompt_specific_guidance: str = "", additional_guidance: str = "") -> Dict[str, Any]:
            Validate the generated quiz questions for quality and accuracy.
    """

    def __init__(self, lesson_no: int, llm, course_name: str, lesson_loader: LessonLoader,
                 output_dir: Union[Path, str] = None, verbose: bool = False, slide_dir: Union[Path, str] = None):
        super().__init__(lesson_no=lesson_no, course_name=course_name, lesson_loader=lesson_loader,
                         output_dir=output_dir, verbose=verbose)

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
        self.readings = self._format_readings_for_prompt()  # Adjust reading formatting

        # Initialize chain and validator
        self.prompt = self._generate_prompt()
        parser = StrOutputParser()
        self.chain = PromptTemplate.from_template(self.prompt) | self.llm | parser
        self.validator = Validator(llm=self.llm, parser=JsonOutputParser(pydantic_object=ValidatorResponse), log_level=self.logger.level)

        # Verify the Beamer file from the previous lesson
        self.beamer_example = self.lesson_loader.find_prior_beamer_presentation(self.lesson_no)
        self.beamer_output = self.output_dir / f'L{self.lesson_no}.tex'

    def _format_readings_for_prompt(self) -> str:
        """
        Formats readings as a single string for use in the LLM prompt.
        Combines readings across all lessons in the specified range.
        """
        all_readings_dict = self._load_readings(self.lesson_no)
        combined_readings = "\n\n".join(f"Lesson {lesson}: {', '.join(readings)}"
                                        for lesson, readings in all_readings_dict.items())
        return combined_readings

    def _find_prior_lesson(self, lesson_no: int, max_attempts: int = 3) -> Path:
        """
        Dynamically finds the most recent prior lesson to use as a template for slide creation.

        Args:
            lesson_no (int): The current lesson number.
            max_attempts (int): The maximum number of previous lessons to attempt loading (default 3).

        Returns:
            Path: The path to the found Beamer file from a prior lesson.

        Raises:
            FileNotFoundError: If no valid prior lesson file is found within the `max_attempts` range.
        """
        for i in range(1, max_attempts + 1):
            prior_lesson = lesson_no - i
            beamer_file = self.lesson_loader.slide_dir / f'L{prior_lesson}.tex'

            # Check if the Beamer file exists for this prior lesson
            if beamer_file.is_file():
                self.logger.info(f"Found prior lesson: Lesson {prior_lesson}")
                return beamer_file

        # Raise error if no valid prior Beamer file is found within the attempts
        raise FileNotFoundError(f"No prior Beamer file found within the last {max_attempts} lessons.")

    def _load_prior_lesson(self) -> str:
        """
        Load the previous lesson's Beamer presentation as a string.
        """
        return self.lesson_loader.load_beamer_presentation(self.beamer_example)

    def _generate_prompt(self) -> str:
        """
        Generate the LLM prompt based on the lesson objectives, readings, and prior lesson content.

        Args:
            objectives_text (str): The objectives for the current lesson.
            combined_readings_text (str): The combined readings for the lesson.
            prior_lesson (str): The previous lesson's LaTeX content.
            course_name (str): The name of the course being instructed

        Returns:
            str: The prompt string to be sent to the LLM.
        """

        prompt = f"""
        You are a LaTeX Beamer specialist and a political scientist with expertise in {self.course_name}.
        You will be creating the content for a college-level lesson based on the following texts and objectives.
        We are on lesson {self.lesson_no}. Here are the objectives for this lesson.
        ---
        {{objectives}}
        ---
        Here are the texts for this lesson:
        ---
        {{information}}.
        ---
        ### General Format to follow:
          - Each slide should have a title and content, with the content being points that work toward the lesson objectives.
          - The lessons should always include a slide placeholder for a student current event presentation after the title page,
              then move on to where we are in the course, what we did last lesson (Lesson {self.lesson_no - 1}),
              and the lesson objectives for that day. The action in each lesson objective should be bolded (e.g. '\\textbf(Understand) the role of government.')
          - After that we should include a slide with an open-ended and thought-provoking discussion question relevant to the subject matter.
          - The slides should conclude with the three primary takeaways from the lesson, hitting on the lesson points students should remember the most.

        This lesson specifically should discuss:
        ---
        {{specific_guidance}}
        ---
        One slide should also include an exercise the students might engage in to help with their learning.
        This exercise should happen in the middle of the lesson, to get students re-energized.

        Use the prior lesson’s presentation as an example:
        ---
        {{last_presentation}}
        ---
        {{additional_guidance}}
        ---
        ### IMPORTANT:
         - You **must** strictly follow the LaTeX format. Your response should **only** include LaTeX code without any extra explanations.
         - Start your response at the point in the preamble where we call `\\title`.
         - Ensure the LaTeX code is valid, and do not include additional text outside the code blocks.
         - Failure to follow this will result in the output being rejected.

        ### Example of Expected Output:
            \\title{{{{Lesson 5: Interest Groups}}}}
            \\begin{{{{document}}}}
            \\maketitle
            \\section{{{{Lesson Overview}}}}
            \\begin{{{{frame}}}}
            \\titlepage
            \\end{{{{frame}}}}
            ...
            \\end{{{{document}}}}
        """
        return prompt

    def generate_slides(self, specific_guidance: str = None, latex_compiler: str = "pdflatex") -> str:
        """
        Generate the Beamer slides for the lesson using the language model (LLM).

        Args:
            specific_guidance (Optional[str]): Specific guidance for the lesson content. Defaults to None.
            latex_compiler (str): The full path or name of the LaTeX compiler executable if it's not on the PATH.

        Returns:
            str: Generated LaTeX content for the slides.
        """
        # Load objectives, readings, and previous lesson slides
        objectives_text = self.lesson_loader.extract_lesson_objectives(self.lesson_no) if not self.user_objectives else self.user_objectives
        combined_readings_text = self.readings

        if self.lesson_loader.slide_dir:
            prior_lesson = self._load_prior_lesson()
        else:
            prior_lesson = "Not Provided"
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
            response = self.chain.invoke({
                "objectives": objectives_text,
                "information": combined_readings_text,
                "last_presentation": prior_lesson,
                'specific_guidance': specific_guidance if specific_guidance else "Not provided.",
                "additional_guidance": additional_guidance
            })

            val_response = self._validate_llm_response(generated_slides=response,
                                                       objectives=objectives_text,
                                                       readings=combined_readings_text,
                                                       last_presentation=prior_lesson,
                                                       prompt_specific_guidance=specific_guidance if specific_guidance else "Not provided.")

            # Validate raw LLM response for quality
            self.validator.logger.info(f"Validation output: {val_response}")
            if int(val_response['status']) == 1:
                valid = True
            else:
                retries += 1
                additional_guidance = val_response.get("additional_guidance", "")
                self.validator.logger.warning(f"Response validation failed on attempt {retries}. "
                                              f"Guidance for improvement: {additional_guidance}")

            # Clean and format the LaTeX output
            cleaned_latex = clean_latex_content(response)
            full_latex = preamble + "\n\n" + comment_out_includegraphics(cleaned_latex)
            self.llm_response = full_latex

            # Validate the generated LaTeX code
            is_valid_latex = False  # Reset each iteration
            try:
                is_valid_latex = validate_latex(full_latex, latex_compiler=latex_compiler)
            except Exception as e:
                self.logger.error(f"LaTeX validation encountered an error: {e}")

            if is_valid_latex:
                valid = True
            else:
                retries += 1  # Increment retries only if validation fails
                self.logger.warning("\nLaTeX code is invalid. Attempting a second model run. "
                                    "If the error persists, please review the LLM output for potential causes. "
                                    "You can inspect the model output via the 'llm_response' object (BeamerBot.llm_response). "
                                    "\n\nNote: Compilation issues may stem from syntax errors in the example LaTeX code provided to the model."
                                    )

        # Handle validation failure after max retries
        if not valid:
            raise ValueError("Validation failed after max retries. Ensure correct prompt and input data. Consider trying a different LLM.")

        return full_latex

    def _validate_llm_response(self, generated_slides: str, objectives: str, readings: str, last_presentation: str,
                               prompt_specific_guidance: str = "", additional_guidance: str = "") -> Dict[str, Any]:
        """
        Validate the generated quiz questions by sending them to the validator for quality and accuracy checks.

        Args:
            quiz_questions (Dict[str, Any]): The generated quiz questions from the LLM, structured as a dictionary.
            objectives (str): Lesson objectives to provide context for validation.
            readings (str): Text content of the lesson readings, used to evaluate the relevance of generated questions.
            last_presentation (str): The prior presentation used as format guide for the new presentation.
            specific_guidance (Optional, str): Manual prompt adjustments passed by the user at runtime.
            additional_guidance (Optional,str): Extra guidance provided to the validator to improve response accuracy.

        Returns:
            Dict[str, Any]: A dictionary containing the validation response, which includes fields such as
                            "evaluation_score," "status," "reasoning," and "additional_guidance."
        """
        # Validate quiz quality and accuracy
        val_template = PromptTemplate.from_template(self.prompt)
        response_str = str(generated_slides)
        validation_prompt = val_template.format(course_name=self.course_name,
                                                objectives=objectives,
                                                information=readings,
                                                last_presentation=last_presentation,
                                                additional_guidance=additional_guidance,
                                                specific_guidance=prompt_specific_guidance
                                                )
        val_response = self.validator.validate(task_description=validation_prompt,
                                               generated_response=response_str,
                                               min_eval_score=8,
                                               specific_guidance="Pay attention to the concepts introduced and their accuracy with respect to the texts.")

        return val_response

    def save_slides(self, latex_content: str):
        """
        Save the generated LaTeX content to a .tex file.

        Args:
            latex_content (str): The LaTeX content to save.
        """
        with open(self.beamer_output, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        self.logger.info(f"Slides saved to {self.beamer_output}")


if __name__ == "__main__":
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI

    from class_factory.utils.tools import reset_loggers

    wd = here()
    load_dotenv()

    user_home = Path.home()

    reset_loggers(log_level=logging.INFO)

    OPENAI_KEY = os.getenv('openai_key')
    OPENAI_ORG = os.getenv('openai_org')

    # Paths for readings, slides, and syllabus
    reading_dir = user_home / os.getenv('readingsDir')
    slide_dir = user_home / os.getenv('slideDir')
    syllabus_path = user_home / os.getenv('syllabus_path')

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_KEY,
        organization=OPENAI_ORG,
    )

    lsn = 12

    # llm = Ollama(
    #     model="llama3.1",
    #     temperature=0.2
    # )

    specific_guidance = """
    The lesson should be structured in a way that discusses big picture ideas about political parties and their influence.
    The slides will, at a minimum, cover the following:
               - The role of parties in government and society
               - How parties have changed over time
               - Relating parties to Tocqueville's notion of associations
               - Have a slide for each of the 5 functions of parties: Recruit Candidates​, Nominate Candidates​, Get Out the Vote (GOTV)​, Facilitate Electoral Choice, Influence National Government​
    """

    loader = LessonLoader(syllabus_path=syllabus_path,
                          reading_dir=reading_dir,
                          slide_dir=slide_dir)

    # Initialize the BeamerBot
    beamer_bot = BeamerBot(
        lesson_no=lsn,
        lesson_loader=loader,
        llm=llm,
        course_name="American Government",
        verbose=True
    )

    # Generate slides for Lesson 20
    generated_slides = beamer_bot.generate_slides()

    # Save the generated LaTeX slides
    # beamer_bot.save_slides(generated_slides)
