"""
**BeamerBot Module**
--------------------

The `BeamerBot` module provides a framework for generating structured LaTeX Beamer slides based on lesson objectives, readings, and prior lesson presentations. By using a language model (LLM), `BeamerBot` automates the process of slide creation, ensuring a consistent slide structure while allowing for custom guidance and validation.

Key Functionalities
~~~~~~~~~~~~~~~~~~~

1. **Automated Slide Generation**:
   - `BeamerBot` generates a LaTeX Beamer presentation for each lesson, incorporating:
     - A title page with consistent author and institution information
     - "Where We Came From" and "Where We Are Going" slides
     - Lesson objectives with highlighted action verbs (e.g., `\\textbf{Analyze} key events`)
     - Discussion questions and in-class exercises
     - Summary slides with key takeaways

2. **Previous Lesson Integration**:
   - Retrieves and references prior lesson presentations to maintain consistent formatting and flow
   - Preserves author and institution information across presentations

3. **Prompt Customization and Validation**:
   - Supports custom prompts and specific guidance for tailored slide content
   - Validates generated LaTeX for correct formatting and content quality
   - Provides multiple retry attempts if validation fails

Dependencies
~~~~~~~~~~~~~

This module requires:

- `langchain_core`: For LLM chain creation and prompt handling
- `pathlib`: For file path management
- Custom utility modules for:
  - Document loading (`load_documents`)
  - LaTeX validation (`llm_validator`)
  - Response parsing (`response_parsers`)
  - Slide pipeline utilities (`slide_pipeline_utils`)

Usage
~~~~~~~

1. **Initialize BeamerBot**:
   ```python
   beamer_bot = BeamerBot(
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
   slides = beamer_bot.generate_slides(specific_guidance=guidance)
   ```

3. **Save the Slides**:
   ```python
   beamer_bot.save_slides(slides)
   ```
"""

import logging
from pathlib import Path
from typing import Any, Dict, Union

# env setup
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate)

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
    A class to generate LaTeX Beamer slides for a specified lesson using a language model (LLM).

    BeamerBot automates the slide generation process, creating structured presentations based on lesson
    readings, objectives, and content from prior presentations when available. Each slide is crafted
    following a consistent format, and the generated LaTeX is validated for correctness.

    Attributes:
        lesson_no (int): Lesson number for which to generate slides.
        llm: Language model instance for generating slides.
        course_name (str): Name of the course for slide context.
        lesson_loader (LessonLoader): Loader for accessing lesson readings and objectives.
        output_dir (Path): Directory to save the generated Beamer slides.
        slide_dir (Optional[Path]): Directory containing existing Beamer slides.
        llm_response (str): Stores the generated LaTeX response from the LLM.
        prompt (str): Generated prompt for the LLM.
        lesson_objectives (optional, dict): user-provided lesson objectives if syllabus not available.

    Methods:
        generate_slides(specific_guidance: str = None, latex_compiler: str = "pdflatex") -> str:
            Generates Beamer slides as LaTeX code for the specified lesson.

        save_slides(latex_content: str) -> None:
            Saves the generated LaTeX content to a .tex file.

        set_user_objectives(objectives: Union[List[str], Dict[str, str]]):
            Initialize user-defined lesson objectives, converting lists to dictionaries if needed. Inherited from BaseModel.

    Internal Methods:
        _format_readings_for_prompt() -> str:
            Combines readings across lessons into a single string for the LLM prompt.

        _find_prior_lesson(lesson_no: int, max_attempts: int = 3) -> Path:
            Finds the most recent prior lesson's Beamer file to use as a template.

        _load_prior_lesson() -> str:
            Loads the LaTeX content of a prior lesson's Beamer presentation as a string.

        _generate_prompt() -> str:
            Constructs the LLM prompt using lesson objectives, readings, and prior lesson content.

        _validate_llm_response(generated_slides: str, objectives: str, readings: str, last_presentation: str,
                               prompt_specific_guidance: str = "", additional_guidance: str = "") -> Dict[str, Any]:
            Validates the generated LaTeX for quality and accuracy.
    """

    def __init__(self, lesson_no: int, llm, course_name: str, lesson_loader: LessonLoader,
                 output_dir: Union[Path, str] = None, verbose: bool = False,
                 slide_dir: Union[Path, str] = None, lesson_objectives: dict = None):
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
        self.user_objectives = self.set_user_objectives(lesson_objectives, range(self.lesson_no, self.lesson_no+1)) if lesson_objectives else {}

        # Initialize chain and validator
        self.prompt = self._generate_prompt()
        parser = StrOutputParser()
        self.chain = self.prompt | self.llm | parser
        self.validator = Validator(llm=self.llm, parser=JsonOutputParser(pydantic_object=ValidatorResponse), log_level=self.logger.level)

        # Verify the Beamer file from the previous lesson
        self.prior_lesson = self.lesson_no - 1  # default prior lesson, updated when find prior beamer presentation
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.beamer_output = self.output_dir / f'L{self.lesson_no}.tex'

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
                    f"Lesson {lesson}, Reading {idx}:\n{reading}\n")

        return "\n".join(combined_readings)

    def _find_prior_lesson(self, lesson_no: int, max_attempts: int = 3) -> Path:
        """
        Find the most recent prior lesson's Beamer file to use as a template.

        Args:
            lesson_no (int): The current lesson number.
            max_attempts (int): Number of prior lessons to attempt to retrieve. Defaults to 3.

        Returns:
            Path: The path to the located Beamer file.

        Raises:
            FileNotFoundError: If no valid prior lesson file is found within max_attempts.
        """
        for i in range(1, max_attempts + 1):
            prior_lesson = lesson_no - i
            beamer_file = self.lesson_loader.slide_dir / f'L{prior_lesson}.tex'

            # Check if the Beamer file exists for this prior lesson
            if beamer_file.is_file():
                self.prior_lesson = int(prior_lesson)
                self.logger.info(f"Found prior lesson: Lesson {prior_lesson}")
                return beamer_file

        # Raise error if no valid prior Beamer file is found within the attempts
        raise FileNotFoundError(f"No prior Beamer file found within the last {max_attempts} lessons.")

    def _load_prior_lesson(self) -> str:
        """
        Load the previous lesson's Beamer presentation as a string.
        """
        beamer_example = self.lesson_loader.find_prior_beamer_presentation(self.lesson_no)
        return self.lesson_loader.load_beamer_presentation(beamer_example)

    def _generate_prompt(self, human_prompt: str = None) -> str:
        """
        Generates a detailed prompt for the LLM to guide LaTeX Beamer slide creation.

        Returns:
            str: The constructed prompt for the LLM.
        """

        slide_system_prompt = """You are a LaTeX Beamer specialist and a political scientist with expertise in {course_name}.
        Your task is to create content for a college-level lesson using the Beamer presentation format.
        Focus on clarity, relevance, and adherence to LaTeX standards."""

        slide_human_prompt = """
            ## Create a LaTeX Beamer presentation following the below guidelines:

            ### Source Documents and Examples

            1. **Lesson Objectives**:
               - We are on lesson {lesson_no}.
               - Ensure each slide works toward the following lesson objectives:
               {objectives}

            2. **Lesson Readings**:
               - Use these readings to guide your slide content:
               {information}

            ---

            ### General Format to Follow:

            1. **Title Slide**:
               - Copy the prior lesson's title slide, **include author and institution from the last presentation**.

            2. **Where We Are in the Course**
               - Last time: <Title of last lesson>
                  - The readings from last lesson (Lesson {prior_lesson}).
                  - **Include every assigned reading from this lesson.**
               - Today: <Title of the current lesson>
                  - The readings for the current lesson (Lesson {lesson_no}).
                  - **Include every assigned reading from this lesson**

            3. **Lesson Objectives**:
                - The action in each lesson objective should be bolded (e.g. '\\textbf(Understand) the role of government.')

            4. **Discussion Question**:
               - Add a thought-provoking question based on lesson material to initiate conversation.

            5. **Lecture Slides**:
               - Cover key points from the lesson objectives and readings.
               - Ensure logical flow and alignment with the objectives.

            6. **In-Class Exercise**:
               - Add an interactive exercise to engage and re-energize students.
               - This exercise should occur about halfway through the lecture slides, to get students re-engaged.

            7. **Key Takeaways**:
               - Conclude with three primary takeaways from the lesson. These should emphasize the most critical points.
               - Bold or italicize the key terms for emphasis.

            8. **Next Time**:
               - Provide the title of the next lesson (Lesson {lesson_no + 1}).
               - Include the assigned readings for the next lesson (Lesson {lesson_no + 1}).

            ---

            ### Specific guidance for this lesson:

            {specific_guidance}

            ---

            ### Example of Expected Output:
                % This is an example format only. Use the provided last lesson as your primary source.
                % Replace the example \\author{{}} and \\institute{{}} below with the corresponding values from last lesson's presentation
                \\title{{Lesson 5: Interest Groups}}
                \\author{{}}
                \\institute[]{{}}
                \\date{{\\today}}
                \\begin{{document}}
                \\section{{Introduction}}
                \\begin{{frame}}
                \\titlepage
                \\end{{frame}}
                ...
                \\end{{document}}


            {additional_guidance}

            ---

            ### Example of previous presentation:
            - Use the presentation from last lesson as an example for formatting and structure:
            {last_presentation}

            ---

            ### IMPORTANT:
            - Use valid LaTeX syntax.
            - The output should contain **only** LaTeX code, with no extra explanations.
            - Start at the point in the preamble where we call \\title.
            - Failure to follow the format and style of the last lesson's presentation may result in the output being rejected.
            - Use the **same author and institute** as provided in the last lesson’s presentation. Do not invent new names or institutions. Copy these values exactly from the prior lesson.
            - If unable to identify the author and institute from the last lesson, just leave them blank.
            - Failure to follow these instructions will result in the output being rejected.
            """

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        slide_system_prompt.format(course_name=self.course_name)
                    )
                ),
                HumanMessagePromptTemplate.from_template(slide_human_prompt if not human_prompt else human_prompt),
            ]
        )

        return prompt

    def generate_slides(self, specific_guidance: str = None, lesson_objectives: dict = None, latex_compiler: str = "pdflatex") -> str:
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
        objectives_text = "\n\n".join([self._get_lesson_objectives(lesson) for lesson in range(self.lesson_no - 1, self.lesson_no + 2)])
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
                "last_presentation": self.prior_lesson,
                "lesson_no": self.lesson_no,
                "prior_lesson": int(self.lesson_no) - 1,
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

            if int(val_response['status']) != 1:
                retries += 1
                additional_guidance = val_response.get("additional_guidance", "")
                self.validator.logger.warning(
                    f"Response validation failed on attempt {retries}. "
                    f"Guidance for improvement: {additional_guidance}"
                )
                continue  # Retry LLM generation

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
                return full_latex
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

    def _validate_llm_response(self, generated_slides: str, objectives: str, readings: str, last_presentation: str,
                               prompt_specific_guidance: str = "", additional_guidance: str = "") -> Dict[str, Any]:
        """
        Validates the generated LaTeX slides for content quality and formatting accuracy.

        Args:
            generated_slides (str): LaTeX content generated by the LLM
            objectives (str): Formatted string of lesson objectives for validation
            readings (str): Formatted string of lesson readings for content verification
            last_presentation (str): Content from prior lesson's presentation for format consistency
            prompt_specific_guidance (str, optional): Custom guidance provided during generation
            additional_guidance (str, optional): Supplementary guidance for validation refinement

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
        # Validate quiz quality and accuracy
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
        val_response = self.validator.validate(task_description=validation_prompt,
                                               generated_response=response_str,
                                               min_eval_score=8,
                                               specific_guidance="Pay attention to the concepts introduced and their accuracy with respect to the texts.")

        return val_response

    def save_slides(self, latex_content: str, output_dir: Union[Path, str] = None) -> None:
        """
        Save the generated LaTeX content to a .tex file.

        Args:
            latex_content (str): The LaTeX content to save.
        """
        if output_dir:
            beamer_file = Path(output_dir) / f'L{self.lesson_no}.tex'
            with open(beamer_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
        else:
            with open(self.beamer_output, 'w', encoding='utf-8') as f:
                f.write(latex_content)
        self.logger.info(f"Slides saved to {self.beamer_output}")


if __name__ == "__main__":
    import os

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

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-1.5-flash-8b",
    #     temperature=0.4,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     api_key=GEMINI_KEY
    # )

    lsn = 3

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
        lesson_no=2,
        lesson_loader=loader,
        llm=llm,
        course_name="American Government",
        verbose=True
    )

    # Generate slides for Lesson 20
    slides = beamer_bot.generate_slides(lesson_objectives={"3": "do nothing today"})

    # Save the generated LaTeX slides
    # beamer_bot.save_slides(slides)

# %%
