"""
Document Loading and Processing Module
-------------------------------------

This module provides functionality to load, process, and extract text from various document types
(PDF, DOCX, and TXT) for generating lesson-specific content. It includes support for OCR processing
of scanned documents and handling of structured educational materials like syllabi and lesson readings.

Classes
~~~~~~~

LessonLoader
    Main class for handling document loading and processing operations.

Key Functions
~~~~~~~

The LessonLoader class provides these key functionalities:

- Document Loading:
    - load_directory: Load all documents from a specified directory
    - load_lessons: Load lessons from multiple directories with lesson number inference
    - load_readings: Extract text from individual documents
    - load_beamer_presentation: Load Beamer presentation content

- Syllabus Processing:
    - extract_lesson_objectives: Extract objectives for specific lessons
    - load_docx_syllabus: Load and parse DOCX syllabus content
    - find_docx_indices: Locate lesson sections within syllabus

- Text Extraction:
    - extract_text_from_pdf: Extract text from PDF files
    - ocr_pdf: Perform OCR on scanned documents
    - convert_pdf_to_docx: Convert PDF files to DOCX format

Dependencies
~~~~~~~~~~~

Core Dependencies:
    - pypdf: PDF text extraction
    - python-docx: DOCX file handling
    - pathlib: File path operations
    - typing: Type hints

Optional OCR Dependencies:
    - pytesseract: OCR processing
    - pdf2image: PDF to image conversion
    - spacy: Text processing
    - contextualSpellCheck: Text correction
    - img2table: Table extraction from images

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

    from class_factory.utils.load_documents import LessonLoader

    # Initialize loader with paths
    loader = LessonLoader(
        syllabus_path="path/to/syllabus.docx",
        reading_dir="path/to/readings"
    )

    # Load specific lesson content
    lesson_content = loader.load_lessons(lesson_number=5)

    # Extract lesson objectives
    objectives = loader.extract_lesson_objectives(current_lesson=5)

Notes
~~~~~

- OCR functionality requires additional package installation via `pip install class_factory[ocr]`
- Directory structure should follow consistent naming (e.g., 'L1', 'L2', etc.)
- Supports both PDF and DOCX syllabus formats with automatic conversion if needed

See Also
~~~~~~~~

- :class:`class_factory.utils.tools.logger_setup`: Logger configuration
- :mod:`class_factory.utils.base_model`: Base model implementation
"""
# %%
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pypdf
from docx import Document
# doc import
from docx.opc.exceptions import PackageNotFoundError
from markitdown import (FileConversionException, MarkItDown,
                        UnsupportedFormatException)
from pdf2docx import Converter
from pyprojroot.here import here

from class_factory.utils.tools import logger_setup

try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None
    convert_from_path = None
    spacy = None
    contextualSpellCheck = None
    img2table = None


class LessonLoader:
    """
    A class for loading and managing educational content from various document formats.

    This class handles loading and processing of lesson materials including syllabi, readings,
    and Beamer presentations. It supports multiple file formats (PDF, DOCX, TXT) and provides
    OCR capabilities for scanned documents when necessary.

    Attributes:
        reading_dir (Path): Directory containing lesson reading materials
        slide_dir (Path): Directory containing Beamer presentation slides
        project_dir (Path): Root project directory
        syllabus_path (Path): Path to the course syllabus file
        logger (Logger): Class logger instance

    Args:
        syllabus_path (Union[Path, str]): Path to the syllabus file
        reading_dir (Union[Path, str]): Directory containing lesson readings
        slide_dir (Union[Path, str], optional): Directory for Beamer slides. Defaults to None.
        project_dir (Union[Path, str], optional): Root directory for the project. Defaults to None.
        verbose (bool, optional): Whether to show detailed logging. Defaults to True.
    """

    def __init__(self, syllabus_path: Union[Path, str], reading_dir: Union[Path, str],
                 slide_dir: Union[Path, str, None] = None, project_dir: Union[Path, str, None] = None,
                 verbose: bool = True, tabular_syllabus: bool = False):
        """
        Initializes LessonLoader with paths and performs validation.

        Args:
            syllabus_path (Union[Path, str]): Path to the syllabus file.
            reading_dir (Union[Path, str]): Directory for lesson readings.
            slide_dir (Union[Path, str]): Directory for Beamer slides, Default is None.
            project_dir (Union[Path, str]): Root directory for the project, Default is None. If not provided, returns pyprojroot.here.here()
            log_level (int): Logging level for the LessonLoader instance.
        """
        log_level = logging.INFO if verbose else logging.WARNING
        self.logger = logger_setup(logger_name="lesson_loader_logger", log_level=log_level)
        self.reading_dir = self._validate_dir_path(reading_dir, "reading directory")
        self.slide_dir = self._validate_dir_path(slide_dir, "slide directory") if slide_dir else None
        self.project_dir = self._validate_dir_path(project_dir, "root project directory") if project_dir else here()
        self.syllabus_path = self._validate_file_path(syllabus_path, "syllabus file") if syllabus_path else None

        if not syllabus_path:
            self.logger.warning("No syllabus path provided. You can manually set objectives, but syllabus-based extraction is recommended.")

        self.tabular_syllabus = tabular_syllabus

    @property
    def slide_dir(self):
        return self._slide_dir

    @slide_dir.setter
    def slide_dir(self, value):
        """Update slide_dir and validate if a path is provided."""
        if value is not None:
            self._slide_dir = self._validate_dir_path(value, "slide directory")
            self.logger.info(f"Slide directory updated to {self._slide_dir}")
        else:
            self._slide_dir = None  # Clear if None is passed

    def _validate_directory_structure(self):
        """
        Validate the directory structure for lesson readings, ensuring that each directory
        follows the expected pattern and contains supported file types.

        Expected Directory Naming:
            Each directory should correspond to a lesson and follow a naming pattern such as:
            - "L1", "Lesson_1", "Lecture 1", "W1", "Week_1", etc.
            - The pattern allows optional spaces or underscores between the lesson identifier (e.g., "L", "Lesson", etc.)
              and the lesson number.

        Supported File Types:
            Each lesson directory should contain at least one reading file of type: .pdf, .txt, or .docx.

        Raises:
            ValueError: If the structure does not match the expected layout, listing any issues found.
        """
        # Updated regex pattern to allow for flexible directory names with optional spaces or underscores
        expected_pattern = r'^(L|Lesson|Lecture|W|Week)[\s_-]*\d+$'
        issues = []

        for subdir in self.reading_dir.iterdir():
            if subdir.is_dir():
                if not re.match(expected_pattern, subdir.name, re.IGNORECASE):
                    issues.append(f"Directory '{subdir.name}' does not match expected pattern '{expected_pattern}'. "
                                  "Recommend consistent naming convention by lesson (e.g., 'L1', 'L2', etc.).")

                # Check for supported file formats within each lesson directory
                if not any(file.suffix in ['.pdf', '.txt', '.docx'] for file in subdir.iterdir() if file.is_file()):
                    issues.append(f"Directory '{subdir.name}' does not contain any supported reading files (.pdf, .txt, .docx).")

        if issues:
            self.logger.warning("Directory structure validation failed with the following issues:\n" + "\n".join(issues) +
                                "\nContinue if desired or this is a known issue in the provided directory.")
            # raise ValueError("Directory structure validation failed with the following issues:\n" + "\n".join(issues))

    @staticmethod
    def _validate_file_path(path: Union[Path, str], name: str) -> Path:
        """
        Validates that the given path is an existing file.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"The {name} at path '{path}' does not exist or is not a file.")
        return path

    @staticmethod
    def _validate_dir_path(path: Union[Path, str], name: str, create_if_missing: bool = False) -> Path:
        """
        Validates that the given path is an existing directory. Optionally creates it if missing.

        Args:
            path (Union[Path, str]): The directory path to validate.
            name (str): Directory name for error messages.
            create_if_missing (bool): If True, creates the directory if it doesn't exist.
        """

        path = Path(path)
        if not path.exists() and create_if_missing:
            path.mkdir(parents=True)
        elif not path.is_dir():
            raise NotADirectoryError(f"The {name} at path '{path}' does not exist or is not a directory.")
        return path

    @staticmethod
    def ocr_available():
        """Validate if current packages installed support OCR"""
        return all([pytesseract, Image, convert_from_path, spacy, contextualSpellCheck, img2table])

    @staticmethod
    def missing_ocr_packages():
        packages = {
            "pytesseract": pytesseract,
            "pillow": Image,
            "pdf2image": convert_from_path,
            "spacy": spacy,
            "contextualSpellCheck": contextualSpellCheck,
            "img2table": img2table
        }
        return [pkg_name for pkg_name, module in packages.items() if module is None]

    def load_readings(self, file_path: Union[str, Path]) -> str:
        """
        Load text content from a single document file.

        Args:
            file_path (Union[str, Path]): Path to the document file to load

        Returns:
            str: Extracted text content prefixed with the file title

        Raises:
            ValueError: If file type is unsupported or file is corrupted
            ImportError: If OCR packages are needed but not installed
        """
        file_path = Path(file_path)
        text = 'title: ' + file_path.stem + "\n"
        try:
            if file_path.suffix.lower() == '.pdf':
                extracted_text = self.extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                try:
                    doc = Document(str(file_path))
                    extracted_text = "\n".join([para.text for para in doc.paragraphs])
                except PackageNotFoundError:
                    raise ValueError(f"Unable to open {file_path.name}. The file might be corrupted.")
            elif file_path.suffix.lower() == '.txt':
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        extracted_text = file.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='ISO-8859-1') as file:
                        extracted_text = file.read()
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            if not extracted_text.strip():
                self.logger.warning(f"No readable text found in {file_path.name}. File may be empty or unreadable.")

        except Exception as e:
            self.logger.error(f"An error occurred while reading {file_path.name}: {e}")
            raise e

        return text + extracted_text if extracted_text.strip() else "No readable text found."

    def load_directory(self, load_from_dir: Union[Path, str]) -> List[str]:
        """
        Load all valid document files within a directory.

        Args:
            load_from_dir (Union[Path, str]): Directory path to load documents from

        Returns:
            List[str]: List of extracted text content from all valid documents
        """
        load_from_dir = Path(load_from_dir)

        all_documents = []
        for file in load_from_dir.glob('*'):
            if file.suffix in ['.pdf', '.txt', '.docx']:
                all_documents.append(self.load_readings(file))
        return all_documents

    def load_lessons(self, lesson_number_or_range: Union[int, range], logger=None) -> Dict[str, List[str]]:
        """
        Load specific lessons by scanning directories based on lesson numbers.

        Args:
            lesson_number_or_range (Union[int, range]): A single lesson number or range of lesson numbers to load.
            logger: Optional custom logger.

        Returns:
            Dict[str, List[str]]: A dictionary where each key is a lesson number and each value is a list of readings.
        """
        logger = logger or logger_setup(logger_name="load_lessons_func_logger", log_level=logging.WARNING)
        lesson_range = (range(lesson_number_or_range, lesson_number_or_range + 1)
                        if isinstance(lesson_number_or_range, int)
                        else lesson_number_or_range)

        all_readings = {}
        # same pattern as validate_directory_structure
        expected_pattern = r'^(L|Lesson|Lecture|W|Week)[\s_-]*(\d+)$'

        filenames = []
        subdirs = []
        for subdir in self.reading_dir.iterdir():
            if subdir.is_dir():
                # load actual files
                match = re.match(expected_pattern, subdir.name, re.IGNORECASE)
                if match:
                    lesson_no = int(match.group(2))
                    if lesson_no in lesson_range:
                        logger.info(f"Loading readings for lesson {lesson_no} from directory: {subdir.name}")
                        all_readings[str(lesson_no)] = self.load_directory(subdir)

                        # load filenames for user
                        for file in subdir.glob('*'):
                            if file.suffix in ['.pdf', '.txt', '.docx']:
                                filenames.append(file.name)
                                subdirs.append(subdir.name)
                    else:
                        logger.info(f"Skipping directory {subdir.name} as it is not in the specified lesson range.")
                else:
                    logger.warning(f"Directory '{subdir.name}' does not match expected lesson naming pattern.")

        pretty_subdirs = ", ".join(list(dict.fromkeys(subdirs))) if subdirs else "(none)"
        if filenames:
            pretty = "\n  - " + "\n  - ".join(filenames)
            self.logger.info(
                f"""Lesson {lesson_number_or_range}: using the following readings:\n{pretty}
                    \nIf this is incorrect, please add the correct readings to the respective lesson folder(s): {pretty_subdirs}."""
            )
        else:
            self.logger.warning(
                f"""Lesson {lesson_number_or_range}: no readings assigned. If this is unexpected, please add the correct readings to the respective lesson folder(s): {
                    pretty_subdirs}."""
            )

        return all_readings

    def load_beamer_presentation(self, tex_path: Path) -> str:
        """
        Loas a Beamer presentation from a .tex file and returns it as a string.

        Args:
            tex_path (Path): The path to the .tex file containing the Beamer presentation.
        Returns:
            str: The content of the .tex file.
        """
        if tex_path == "No prior presentation available.":
            return tex_path

        tex_path = Path(tex_path)
        with open(tex_path, 'r', encoding='utf-8') as file:
            beamer_text = file.read()
        return beamer_text

    def find_prior_beamer_presentation(self, lesson_no: int, max_attempts: int = 3) -> Path | str:
        """
        Dynamically finds the most recent prior lesson to use as a template for slide creation.

        Args:
            lesson_no (int): The current lesson number.
            max_attempts (int): The maximum number of previous lessons to attempt loading (default 3).

        Returns:
            Path | str: The path to the found Beamer file from a prior lesson.

        Raises:
            FileNotFoundError: If no valid prior lesson file is found within the `max_attempts` range.
        """
        for i in range(1, max_attempts + 1):
            prior_lesson = lesson_no - i
            beamer_file = self.slide_dir / f'L{prior_lesson}.tex' if self.slide_dir else Path(f'L{prior_lesson}.tex')

            # Check if the Beamer file exists for this prior lesson
            if beamer_file.is_file():
                self.logger.info(f"Found prior lesson: Lesson {prior_lesson}")
                return beamer_file

        # Raise error if no valid prior Beamer file is found within the attempts
        self.logger.error(f"No prior Beamer file found within the last {max_attempts} lessons.\nReturning empty.")
        return "No prior presentation available."

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text content from a PDF file, with OCR fallback if needed.

        Args:
            pdf_path (Union[str, Path]): Path to the PDF file

        Returns:
            str: Extracted text content from the PDF

        Raises:
            ImportError: If text extraction fails and OCR packages are not available
        """
        text_content = []
        pdf_path = Path(pdf_path)
        with open(str(pdf_path), 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    paragraphs = page_text.split('\n\n')
                    text_content.extend(paragraph.strip() for paragraph in paragraphs if paragraph.strip())
        if not text_content and self.ocr_available():
            self.logger.warning(f"No readable text found in pdf {pdf_path.name}. Attempting OCR.")
            ocr_result = self.ocr_pdf(pdf_path, max_workers=6)
            return ocr_result if ocr_result.strip() else "OCR did not produce readable text."
        elif not text_content and not self.ocr_available():
            missing = self.missing_ocr_packages()
            raise ImportError(
                f"No readable text found in PDF {pdf_path.name}.\nOCR support requires: {', '.join(missing)}."
                "\nInstall with `pip install class_factory[ocr]`, or convert the file to readable text."
            )
        return ' '.join(text_content)

    def ocr_pdf(self, pdf_path: Path, max_workers: int = 4) -> str:
        """
        Perform OCR on a PDF file to extract text content.

        Args:
            pdf_path (Path): Path to the PDF file
            max_workers (int, optional): Number of parallel workers for OCR. Defaults to 4.

        Returns:
            str: Extracted text content from OCR
        """
        import pytesseract
        from pdf2image import convert_from_path

        pdf_path = Path(pdf_path)
        images = convert_from_path(str(pdf_path), dpi=300)
        ocr_text = []
        for image in images:
            text = pytesseract.image_to_string(image)
            ocr_text.append(text)
        return " ".join(ocr_text)

    def load_docx_syllabus(self, syllabus_path) -> List[str]:
        max_retries = 3
        retry_delay = 10  # seconds
        syllabus_path = Path(syllabus_path)
        md = MarkItDown()
        for attempt in range(max_retries):
            try:
                raw_content = md.convert(str(syllabus_path))
                lines = raw_content.text_content.split("\n\n")
                for line in lines:
                    # Handle table-like rows in lesson objectives
                    if line.startswith("|"):
                        tablines = line.split("\n")
                        lines.remove(line)
                        lines.extend(tablines)

                return lines

            except (PackageNotFoundError, FileConversionException, UnsupportedFormatException) as e:
                if attempt < max_retries - 1:
                    print(f"Document `{syllabus_path.name}` is currently open. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise PackageNotFoundError("Unable to open the document after multiple attempts. Please close the file and try again.")

    def extract_lesson_objectives(self, current_lesson: Union[int, str], only_current: bool = False,
                                  ) -> str:
        """
        Extract lesson objectives from the syllabus for specified lesson(s).

        Args:
            current_lesson (Union[int, str]): The lesson number to extract objectives for
            only_current (bool, optional): If True, return only current lesson objectives.
                If False, include previous and next lessons. Defaults to False.

        Returns:
            str: Extracted lesson objectives text. Returns "No lesson objectives provided"
                if no syllabus path is set.
        """
        if not self.syllabus_path:
            return "No lesson objectives provided."
        syllabus_path = Path(self.syllabus_path)
        # if syllabus_path.suffix == '.pdf':
        #     syllabus_path = self.convert_pdf_to_docx(syllabus_path)
        syllabus_content = self.load_docx_syllabus(syllabus_path)

        current_lesson = int(current_lesson)
        prev_idx, curr_idx, next_idx, end_idx = self.find_docx_indices(
            syllabus_content, current_lesson)

        if self.tabular_syllabus:
            prev_lesson_content = syllabus_content[prev_idx] if prev_idx is not None else ""
            curr_lesson_content = syllabus_content[curr_idx] if curr_idx is not None else ""
            next_lesson_content = syllabus_content[next_idx] if next_idx is not None else ""

        else:
            prev_lesson_content = "\n".join(
                syllabus_content[prev_idx:curr_idx]) if prev_idx is not None else ""
            curr_lesson_content = "\n".join(
                syllabus_content[curr_idx:next_idx]) if curr_idx is not None else ""
            next_lesson_content = "\n".join(
                syllabus_content[next_idx:end_idx]) if next_idx is not None else ""

        combined_content = "\n\n".join(filter(None, [prev_lesson_content, curr_lesson_content, next_lesson_content]))
        return curr_lesson_content if only_current else combined_content

    def find_docx_indices(self, syllabus: List[str], current_lesson: int, lesson_identifier: str = ""
                          ) -> Tuple[int | None, int | None, int | None, int | None]:
        """
        Finds the indices of the lessons in the syllabus content.

        Args:
            syllabus (List[str]): A list of strings where each string represents a line in the syllabus document.
            current_lesson (int): The lesson number for which to find surrounding lessons.
            lesson_identifier (str, Defaults to None): The special word indicating a new lesson on the syllabus (eg "Lesson" or "Week")
        Returns:
            Tuple[int, int, int, int]: The indices of the previous, current, next, and the end of the next lesson.
        """
        prev_lesson, curr_lesson, next_lesson, end_lesson = None, None, None, None

        if self.tabular_syllabus:
            # Search tables if no matches found
            if curr_lesson is None:
                # Matches table rows starting with numbers or "|"
                lesson_pattern = re.compile(r"^(\d+)\s*\||^\s*\|\s*(\d+)\s*\|")
                for i, line in enumerate(syllabus):
                    match = lesson_pattern.match(line)
                    if match:
                        matches = match.groups()
                        # Extract numeric lesson from match groups
                        lesson_number = int(matches[0] or matches[1])
                        if lesson_number == current_lesson - 1:
                            prev_lesson = i
                        elif lesson_number == current_lesson:
                            curr_lesson = i
                        elif lesson_number == current_lesson + 1:
                            next_lesson = i
                        elif lesson_number == current_lesson + 2:
                            end_lesson = i
                           # break
            return prev_lesson, curr_lesson, next_lesson, end_lesson

        else:
            if not lesson_identifier:
                lesson_identifiers = ['Lesson', 'Week', "**Lesson", "**Week"]

                for lesson_identifier in lesson_identifiers:
                    escaped_identifier = re.escape(lesson_identifier)
                    lesson_pattern = re.compile(
                        rf"{escaped_identifier}\s*{current_lesson}.*?:")

                    for i, line in enumerate(syllabus):
                        if re.search(rf"{escaped_identifier}\s*{current_lesson - 1}.*?:?", line):
                            prev_lesson = i
                        elif lesson_pattern.search(line):
                            curr_lesson = i
                        elif re.search(rf"{escaped_identifier}\s*{current_lesson + 1}.*?:?", line):
                            next_lesson = i
                        elif re.search(rf"{escaped_identifier}\s*{current_lesson + 2}.*?:?", line):
                            end_lesson = i
                            break
                    if curr_lesson is not None:
                        break
            else:
                escaped_identifier = re.escape(lesson_identifier)
                lesson_pattern = re.compile(
                    rf"{escaped_identifier}\s*{current_lesson}.*?:")

                for i, line in enumerate(syllabus):
                    if re.search(rf"{escaped_identifier}\s*{current_lesson - 1}.*?:?", line):
                        prev_lesson = i
                    elif lesson_pattern.search(line):
                        curr_lesson = i
                    elif re.search(rf"{escaped_identifier}\s*{current_lesson + 1}.*?:?", line):
                        next_lesson = i
                    elif re.search(rf"{escaped_identifier}\s*{current_lesson + 2}.*?:?", line):
                        end_lesson = i
                        break
            if curr_lesson is None:
                self.logger.warning(
                    f"Lesson {current_lesson} not found in syllabus. Is this a tabular syllabus? Consider setting `tabular_syllabus=True`.")
            return prev_lesson, curr_lesson, next_lesson, end_lesson

# %%


if __name__ == "__main__":
    import os

    import yaml
    from dotenv import load_dotenv
    user_home = Path.home()
    load_dotenv()
    pdf_syllabus_path = user_home / os.getenv('pdf_syllabus_path', "")

    with open("class_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    class_config = config['PS460']

    slide_dir = user_home / class_config['slideDir']
    syllabus_path = user_home / class_config['syllabus_path']
    readingsDir = user_home / class_config['reading_dir']
    is_tabular_syllabus = class_config['is_tabular_syllabus']

    nontab_syllabus = user_home / config['PS211']['syllabus_path']

    lsn = 8

    loader = LessonLoader(syllabus_path=syllabus_path,
                          reading_dir=readingsDir,
                          slide_dir=slide_dir,
                          tabular_syllabus=is_tabular_syllabus)

    objs = loader.extract_lesson_objectives(
        current_lesson=lsn)
    docs = loader.load_lessons(lesson_number_or_range=range(12, 14))

    ocrDir = Path("C:/Users/Sean/OneDrive - afacademy.af.edu/Documents/Classes/Fall 2024/PS211/02_Class Readings/L21/reference")
    pdf_path = ocrDir / "21.3 Pew Research Center. Beyond Red vs Blue Overview.pdf"
    ocr_result = loader.ocr_pdf(pdf_path)

# %%
