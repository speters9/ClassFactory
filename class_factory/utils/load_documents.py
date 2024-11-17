"""
This module provides functionality to load, process, and extract text from various document types (PDF, DOCX, and TXT)
for the purpose of generating lesson-specific content. It supports operations such as extracting lesson objectives
from syllabi, loading lesson readings from specified directories, and handling various document formats.

Key Functions:
~~~~~~~

1. **load_directory**:
   - Loads all document files from a given directory, specifically targeting files that match the inferred lesson number.
   - Supports PDF, DOCX, and TXT file formats.

2. **load_lessons**:
   - Loads lessons from one or multiple directories, with options for recursive directory search.
   - Provides the ability to infer lesson numbers from either filenames or directory names.

3. **infer_lesson_number**:
   - Infers the lesson number from a directory or file path, based on either the filename or the directory name.

4. **infer_lesson_from_filename**:
   - Extracts the lesson number directly from a filename using regular expressions.

5. **extract_text_from_pdf**:
   - Extracts and returns the text content from a PDF file, handling paragraph breaks appropriately.

6. **extract_lessons_from_page**:
   - Extracts lesson content from a single page of syllabus content, identified by lesson markers.

7. **find_pdf_lessons**:
   - Locates lessons within a PDF syllabus based on the current lesson number and returns the relevant content.

8. **find_docx_indices**:
   - Finds indices for lessons in DOCX syllabi, helping to identify previous, current, and upcoming lessons.

9. **load_docx_syllabus**:
   - Loads the content of a DOCX syllabus and returns it as a list of paragraphs.

10. **extract_lesson_objectives**:
    - Extracts objectives for the specified lesson from either a PDF or DOCX syllabus,
      supporting the retrieval of previous, current, and next lessons' objectives.

11. **load_readings**:
    - Loads text content from PDF, DOCX, or TXT files, prefixing the extracted text with the file's title.
    - Handles potential issues such as unreadable or corrupted files.

Usage
~~~~~~~

This module is primarily designed for applications where structured extraction of lesson materials and objectives
is required, such as in educational content analysis or automated lesson planning systems.

Dependencies
~~~~~~~

- **pypdf**: Used for extracting text from PDF files.
- **python-docx**: Used for handling and extracting text from DOCX files.
- **re**: Regular expressions are used extensively for parsing filenames and directory names to infer lesson numbers.
- **time**: Used to implement retry logic for opening DOCX files.
- **dotenv**: For environment variable management when running the module as a script.

Example
~~~~~~~

The module can be executed as a standalone script to load lesson documents and extract lesson objectives from a specified syllabus.
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pymupdf4llm
import pypdf
from docx import Document
# doc import
from docx.opc.exceptions import PackageNotFoundError
from pdf2docx import Converter
from pyprojroot.here import here

from class_factory.utils.tools import logger_setup

try:
    import contextualSpellCheck
    import img2table
    import pytesseract
    import spacy
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
    def __init__(self, syllabus_path: Union[Path, str], reading_dir: Union[Path, str],
                 slide_dir: Union[Path, str] = None, project_dir: Union[Path, str] = None,
                 verbose: bool = True):
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
            self.logger.warning("No syllabus path provided. You can manually set user objectives in a classfactory class instance. "
                                "However providing an actual syllabus to provide lesson objectives to the LLM is highly recommended")
        # Ensure reading directory is of correct format
        self._validate_directory_structure()

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
            raise ValueError("Directory structure validation failed with the following issues:\n" + "\n".join(issues))

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
        if not path:
            return None
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
        """Load a specific documents. Supported types: .docx, .txt. .pdf"""
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
        """Load all valid document files within a directory."""
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

        for subdir in self.reading_dir.iterdir():
            if subdir.is_dir():
                match = re.match(expected_pattern, subdir.name, re.IGNORECASE)
                if match:
                    lesson_no = int(match.group(2))
                    if lesson_no in lesson_range:
                        logger.info(f"Loading readings for lesson {lesson_no} from directory: {subdir.name}")
                        all_readings[str(lesson_no)] = self.load_directory(subdir)
                    else:
                        logger.info(f"Skipping directory {subdir.name} as it is not in the specified lesson range.")
                else:
                    logger.warning(f"Directory '{subdir.name}' does not match expected lesson naming pattern.")

        return all_readings

    def load_beamer_presentation(self, tex_path: Path) -> str:
        """
        Loas a Beamer presentation from a .tex file and returns it as a string.

        Args:
            tex_path (Path): The path to the .tex file containing the Beamer presentation.
        Returns:
            str: The content of the .tex file.
        """
        tex_path = Path(tex_path)
        with open(tex_path, 'r', encoding='utf-8') as file:
            beamer_text = file.read()
        return beamer_text

    def find_prior_beamer_presentation(self, lesson_no: int, max_attempts: int = 3) -> Path:
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
            beamer_file = self.slide_dir / f'L{prior_lesson}.tex'

            # Check if the Beamer file exists for this prior lesson
            if beamer_file.is_file():
                self.logger.info(f"Found prior lesson: Lesson {prior_lesson}")
                return beamer_file

        # Raise error if no valid prior Beamer file is found within the attempts
        raise FileNotFoundError(f"No prior Beamer file found within the last {max_attempts} lessons.")

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
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
        import pytesseract
        from pdf2image import convert_from_path

        pdf_path = Path(pdf_path)
        images = convert_from_path(str(pdf_path), dpi=300)
        ocr_text = []
        for image in images:
            text = pytesseract.image_to_string(image)
            ocr_text.append(text)
        return " ".join(ocr_text)

    def convert_pdf_to_docx(self, pdf_path: Union[str, Path]) -> Path:
        pdf_path = Path(pdf_path)
        docx_path = pdf_path.with_suffix(".docx")
        cv = Converter(str(pdf_path))
        cv.convert(str(docx_path), start=0, end=None)
        cv.close()
        self.logger.info(f"Successfully converted {pdf_path.name} to .docx")
        return docx_path

    def load_docx_syllabus(self, syllabus_path) -> List[str]:
        max_retries = 3
        retry_delay = 10  # seconds
        syllabus_path = Path(syllabus_path)
        for attempt in range(max_retries):
            try:
                doc = Document(str(syllabus_path))
                return [para.text for para in doc.paragraphs]
            except PackageNotFoundError:
                if attempt < max_retries - 1:
                    print(f"Document `{syllabus_path.name}` is currently open. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise PackageNotFoundError("Unable to open the document after multiple attempts. Please close the file and try again.")

    def extract_lesson_objectives(self, current_lesson: int, only_current: bool = False) -> str:
        if not self.syllabus_path:
            return "No lesson objectives provided."
        syllabus_path = Path(self.syllabus_path)
        if syllabus_path.suffix == '.pdf':
            syllabus_path = self.convert_pdf_to_docx(syllabus_path)
        syllabus_content = self.load_docx_syllabus(syllabus_path)
        prev_idx, curr_idx, next_idx, end_idx = self.find_docx_indices(syllabus_content, current_lesson)
        prev_lesson_content = "\n".join(syllabus_content[prev_idx:curr_idx]) if prev_idx is not None else ""
        curr_lesson_content = "\n".join(syllabus_content[curr_idx:next_idx]) if curr_idx is not None else ""
        next_lesson_content = "\n".join(syllabus_content[next_idx:end_idx]) if next_idx is not None else ""
        combined_content = "\n".join(filter(None, [prev_lesson_content, curr_lesson_content, next_lesson_content]))
        return curr_lesson_content if only_current else combined_content

    def find_docx_indices(self, syllabus: List[str], current_lesson: int, lesson_identifier: str = None) -> Tuple[int, int, int, int]:
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

        if lesson_identifier is None:
            lesson_identifiers = ['Lesson', 'Week']

            for lesson_identifier in lesson_identifiers:
                lesson_pattern = re.compile(rf"{lesson_identifier}\s*{current_lesson}.*?:")

                for i, line in enumerate(syllabus):
                    if re.search(rf"{lesson_identifier}\s*{current_lesson - 1}.*?:?", line):
                        prev_lesson = i
                    elif lesson_pattern.search(line):
                        curr_lesson = i
                    elif re.search(rf"{lesson_identifier}\s*{current_lesson + 1}.*?:?", line):
                        next_lesson = i
                    elif re.search(rf"{lesson_identifier}\s*{current_lesson + 2}.*?:?", line):
                        end_lesson = i
                        break
                if curr_lesson is not None:
                    break
        else:
            lesson_pattern = re.compile(rf"{lesson_identifier}\s*{current_lesson}.*?:")

            for i, line in enumerate(syllabus):
                if re.search(rf"{lesson_identifier}\s*{current_lesson - 1}.*?:?", line):
                    prev_lesson = i
                elif lesson_pattern.search(line):
                    curr_lesson = i
                elif re.search(rf"{lesson_identifier}\s*{current_lesson + 1}.*?:?", line):
                    next_lesson = i
                elif re.search(rf"{lesson_identifier}\s*{current_lesson + 2}.*?:?", line):
                    end_lesson = i
                    break

        return prev_lesson, curr_lesson, next_lesson, end_lesson


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv
    user_home = Path.home()
    load_dotenv()

    slide_dir = user_home / os.getenv('slideDir')

    syllabus_path = user_home / os.getenv('syllabus_path')
    pdf_syllabus_path = user_home / os.getenv('pdf_syllabus_path')
    readingsDir = user_home / os.getenv('readingsDir')

    lsn = 8
    # lsn_objectives_doc = extract_lesson_objectives(syllabus_path,
    #                                                lsn,
    #                                                only_current=True)
    # converted_syllabus_path = convert_pdf_to_docx(pdf_syllabus_path)
    # converted_objectives = extract_lesson_objectives(converted_syllabus_path,
    #                                                  lsn,
    #                                                  only_current=True)
    # lsn_objectives_pdf = extract_lesson_objectives(pdf_syllabus_path,
    #                                                lsn,
    #                                                only_current=True)

    # print(f"doc objectives:\n{lsn_objectives_doc}")
    # print(f"\npdf objectives:\n{lsn_objectives_pdf}")

    # docs = load_lessons(readingsDir, recursive=True, lesson_range=range(20, 21))

    # ocr_test = Path(readingsDir / "L21/21.3 Pew Research Center. Beyond Red vs Blue Overview.pdf")
    # ocr_result = extract_text_from_pdf(ocr_test)
    # print(ocr_result)

    loader = LessonLoader(syllabus_path=syllabus_path,
                          reading_dir=readingsDir,
                          slide_dir=slide_dir)

    objs = loader.extract_lesson_objectives(current_lesson=lsn)
    docs = loader.load_lessons(lesson_number_or_range=range(12, 14))
