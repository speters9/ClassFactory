"""
This script automates the generation of a LaTeX Beamer presentation for a specified lesson
in a course, leveraging lesson readings, course objectives from the syllabus, and a previous
lesson's Beamer presentation as a template.

Workflow:
1. **Git Update Check**:

2. **Lesson Objectives Extraction**:

3. **Readings Aggregation**:

4. **Previous Lesson Integration**:

5. **Prompt Construction**:

6. **LaTeX Generation**:

7. **Saving the Output**:


The script is designed to be run in an environment where all dependencies are available and assumes
that the necessary files (readings, syllabus, previous lesson) are properly organized in directories
specified by environment variables. See Readme for assumed directory structure.


"""


# base libraries
from src.beamer_bot.slide_preamble import preamble
from src.utils.slide_pipeline_utils import (
    clean_latex_content,
    load_beamer_presentation
)
from src.utils.load_documents import load_lessons, extract_lesson_objectives

import os
import logging
import subprocess
import tempfile
from pathlib import Path

# env setup
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.utils.tools import logger_setup
from pyprojroot.here import here
wd = here()

# self-defined utils

load_dotenv()

OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Path definitions
readingDir = Path(os.getenv('readingsDir'))
slideDir = Path(os.getenv('slideDir'))
syllabus_path = Path(os.getenv('syllabus_path'))

# %%


# Function to verify if the reading directory exists
def verify_lesson_dir(lesson_no: int, reading_dir: Path) -> bool:
    """ensure the lesson directory referenced by the user exists"""
    input_dir = reading_dir / f'L{lesson_no}/'
    return input_dir.exists() and any(input_dir.iterdir())  # Check if directory exists and contains files


def verify_beamer_file(beamer_file: Path) -> bool:
    """check to make sure the suggested file actually exists"""
    return beamer_file.exists()


def comment_out_includegraphics(latex_content: str) -> str:
    """
    This function searches for any \includegraphics commands in the LaTeX content
    and comments them out by adding a '%' at the beginning of the line.

    Args:
        latex_content (str): The raw LaTeX content as a string.

    Returns:
        str: The modified LaTeX content with \includegraphics commands commented out.
    """
    return "\n".join(
        ["%" + line if "\\includegraphics" in line else line for line in latex_content.splitlines()]
    )


def validate_latex(latex_code: str) -> bool:
    """
    Validates LaTeX by attempting to compile it using a LaTeX engine.

    Args:
        latex_code (str): The LaTeX code to validate.

    Returns:
        bool: True if LaTeX compiles successfully, False otherwise.
    """
    # Create a temporary directory to save the LaTeX file and compile it
    with tempfile.TemporaryDirectory() as tempdir:
        tex_file = Path(tempdir) / "tempfile.tex"

        # Write the LaTeX code to the temporary .tex file
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(latex_code)

        # Try to compile the LaTeX file using pdflatex
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tempdir,
                timeout=20  # Timeout after 10 seconds
            )

            # Check if compilation was successful
            if result.returncode == 0:
                print("LaTeX compiled successfully!")
                return True
            else:
                print("LaTeX compilation failed!")
                print(result.stdout.decode("utf-8"))
                return False

        except subprocess.TimeoutExpired:
            print("LaTeX compilation timed out!")
            return False


class BeamerBot:
    def __init__(self, lesson_no: int, syllabus_path: Path, reading_dir: Path,
                 slide_dir: Path, llm, output_dir: Path = None, verbose: bool=True):
        """
        Initializes BeamerBot with lesson number, paths, and the LLM instance.

        Args:
            lesson_no (int): Lesson number to generate slides for.
            syllabus_path (Path): Path to the syllabus file.
            reading_dir (Path): Directory where lesson readings are stored.
            slide_dir (Path): Directory where Beamer slides are stored.
            llm: Language model for generating slides.
            output_dir (Path, optional): Directory where the output Beamer slides should be saved. Defaults to None, and slides will be saved to slide_dir
            custom_guidance (str, optional): Custom lesson-specific guidance for the LLM.
        """
        self.lesson_no = lesson_no
        self.syllabus_path = syllabus_path
        self.reading_dir = reading_dir
        self.slide_dir = slide_dir
        self.llm = llm
        self.output_dir = slide_dir if output_dir is None else output_dir
        self.prompt = None

        # setup logging
        log_level = logging.INFO if verbose else logging.WARNING
        self.logger = logger_setup(log_level=log_level)

        # Verify that the reading directory exists for this lesson
        if not verify_lesson_dir(self.lesson_no, self.reading_dir):
            raise FileNotFoundError(f"Lesson {self.lesson_no} readings directory does not exist or is empty.")

        # Verify the Beamer file from the previous lesson
        self.beamer_example = self.find_prior_lesson(self.lesson_no)
        if not verify_beamer_file(self.beamer_example):
            raise FileNotFoundError(f"Beamer file for Lesson {self.lesson_no - 1} does not exist.")

        self.input_dir = self.reading_dir / f'L{self.lesson_no}/'
        self.beamer_output = self.output_dir / f'L{self.lesson_no}.tex'
        self.readings = self.load_readings()

    def load_readings(self) -> str:
        """
        Load the lesson readings from the directory using the standardized `load_lessons` method.

        Returns:
            str: Combined lesson readings as a string.
        """
        return "\n\n".join(load_lessons(self.input_dir, lesson_range=self.lesson_no, recursive=False))

    def find_prior_lesson(self, lesson_no: int, max_attempts: int = 3) -> Path:
        """
        Dynamically finds the most recent prior lesson to use as a template for slide creation.
        If the previous lesson's file does not exist, attempts to load from lessons further back (e.g., 2 or 3 lessons ago).

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
            beamer_file = self.slide_dir / f'L{prior_lesson}.tex'
            if verify_beamer_file(beamer_file):
                self.logger.info(f"Found prior lesson: Lesson {prior_lesson}")
                return beamer_file
        raise FileNotFoundError(f"No prior Beamer file found within the last {max_attempts} lessons.")


    def load_prior_lesson(self) -> str:
        """
        Load the previous lesson's Beamer presentation as a string.

        Returns:
            str: Previous lesson's LaTeX content.
        """
        return load_beamer_presentation(self.beamer_example)

    def generate_prompt(self, objectives_text: str, combined_readings_text: str, prior_lesson: str) -> str:
        """
        Generate the LLM prompt based on the lesson objectives, readings, and prior lesson content.

        Args:
            objectives_text (str): The objectives for the current lesson.
            combined_readings_text (str): The combined readings for the lesson.
            prior_lesson (str): The previous lesson's LaTeX content.

        Returns:
            str: The prompt string to be sent to the LLM.
        """

        prompt = f"""
        You are a LaTeX Beamer specialist and a political scientist with expertise in American politics.
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
          - The slides should conclude with the three primary takeaways from the lesson, hitting on the lesson points they should remember the most.

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
        ### IMPORTANT:
         - You **must** strictly follow the LaTeX format. Your response should **only** include LaTeX code without any extra explanations.
         - Start your response at the point in the preamble where we call `\\title`.
         - Ensure the LaTeX code is valid, and do not include additional text outside the code blocks.
         - Failure to follow this will result in the output being rejected and you not being paid.

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

    def generate_slides(self, specific_guidance: str = None) -> str:
        """
        Generate the Beamer slides for the lesson using the language model (LLM).

        Returns:
            str: Generated LaTeX content for the slides.
        """
        # Load objectives, readings, and previous lesson slides
        objectives_text = extract_lesson_objectives(self.syllabus_path, self.lesson_no)
        combined_readings_text = self.readings
        prior_lesson = self.load_prior_lesson()

        # Prepare the prompt for the language model, save prompt for reference
        self.prompt = self.generate_prompt(objectives_text, combined_readings_text, prior_lesson)

        # Create the LLM prompt chain
        parser = StrOutputParser()
        prompt_template = PromptTemplate.from_template(self.prompt)
        chain = prompt_template | self.llm | parser

       # Generate Beamer slides via the chain
        try:
            response = chain.invoke({
                "objectives": objectives_text,
                "information": combined_readings_text,
                "last_presentation": prior_lesson,
                'specific_guidance': specific_guidance if specific_guidance else "Not provided."
            })
        except Exception as e:
            raise RuntimeError(f"Error during LLM invocation: {e}")

        # Clean and format the LaTeX output
        cleaned_latex = clean_latex_content(response)
        full_latex = preamble + "\n\n" + comment_out_includegraphics(cleaned_latex)

        # Validate the LaTeX code
        is_valid = validate_latex(full_latex)
        assert is_valid, "LaTeX code is invalid. Please review the output."

        return full_latex

    def save_slides(self, latex_content: str):
        """
        Saves the generated LaTeX content to a .tex file.

        Args:
            latex_content (str): The LaTeX content to save.
        """
        with open(self.beamer_output, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        self.logger.info(f"Slides saved to {self.beamer_output}")


if __name__ == "__main__":
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI

    OPENAI_KEY = os.getenv('openai_key')
    OPENAI_ORG = os.getenv('openai_org')

    # Paths for readings, slides, and syllabus
    reading_dir = Path(os.getenv('readingsDir'))
    slide_dir = Path(os.getenv('slideDir'))
    syllabus_path = Path(os.getenv('syllabus_path'))

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
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

    specific_guidance = """
    The lesson should be structured in a way that discusses big picture ideas about political parties and their influence.
    The slides will, at a minimum, cover the following:
               - The role of parties in government and society
               - How parties have changed over time
               - Relating parties to Tocqueville's notion of associations
               - Have a slide for each of the 5 functions of parties: Recruit Candidates​, Nominate Candidates​, Get Out the Vote (GOTV)​, Facilitate Electoral Choice, Influence National Government​
    """

    # Initialize the BeamerBot
    beamer_bot = BeamerBot(
        lesson_no=20,
        syllabus_path=syllabus_path,
        reading_dir=reading_dir,
        slide_dir=slide_dir,
        llm=llm
    )

    # Generate slides for Lesson 20
    generated_slides = beamer_bot.generate_slides(specific_guidance=specific_guidance)

    # Save the generated LaTeX slides
    beamer_bot.save_slides(generated_slides)
