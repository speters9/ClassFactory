"""
This module provides functionality to load, process, and extract text from various document types (PDF, DOCX, and TXT)
for the purpose of generating lesson-specific content. It supports operations such as extracting lesson objectives
from syllabi, loading lesson readings from specified directories, and handling various document formats.

Key Functions:

1. **load_documents**:
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

Usage:
This module is primarily designed for applications where structured extraction of lesson materials and objectives
is required, such as in educational content analysis or automated lesson planning systems.

Dependencies:
- **pypdf**: Used for extracting text from PDF files.
- **python-docx**: Used for handling and extracting text from DOCX files.
- **re**: Regular expressions are used extensively for parsing filenames and directory names to infer lesson numbers.
- **time**: Used to implement retry logic for opening DOCX files.
- **dotenv**: For environment variable management when running the module as a script.

Example:
The module can be executed as a standalone script to load lesson documents and extract lesson objectives from a specified syllabus.
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Tuple, Union

import pypdf
from docx import Document
# doc import
from docx.opc.exceptions import PackageNotFoundError
from pdf2docx import Converter

from class_factory.utils.ocr_pdf_files import ocr_pdf
from class_factory.utils.tools import logger_setup

############################### Lesson loading functions #######################


def load_documents(directory: Path, lesson_number) -> List[str]:
    """
    Load all documents from a single directory.

    Args:
        directory (Path): The directory to search for documents.

    Returns:
        List[str]: A list of document contents as strings.
    """
    all_documents = []

    # Iterate through files in the given directory (no recursion)
    for file in directory.glob('*'):
        if file.suffix in ['.pdf', '.txt', '.docx']:
            inferred_lesson_number = infer_lesson_from_filename(file.name)
            if inferred_lesson_number == lesson_number:
                all_documents.append(load_readings(file))

    return all_documents


def load_lessons(directories: Union[Path, List[Path]], lesson_range: Union[range, int] =
                 None, recursive: bool = True, infer_from: str = "filename") -> List[str]:
    """
    Load specific lessons from one or multiple directories, with options to infer lesson numbers from filenames or directory names.

    Args:
        directories (Union[Path, List[Path]]): The directory or list of directories to search for documents.
        lesson_range (range, optional): The range of lesson numbers to load. If None, all lessons will be loaded.
        recursive (bool): If True, search through all subdirectories recursively (recursion stops one directory deep).
        infer_from (str): Method to infer lesson numbers; options are "filename" or "directory".

    Returns:
        List[str]: A list of document contents as strings.
    """
    logger = logger_setup(log_level=logging.WARNING)
    if isinstance(directories, (str, Path)):
        directories = [Path(directories)]

    # If an integer is provided, convert it to a range of a single lesson
    if isinstance(lesson_range, int):
        lesson_range = range(lesson_range, lesson_range + 1)

    all_documents = []

    for directory in directories:
        if recursive:
            # Use rglob for recursive search in all subdirectories
            subdirectories = [p for p in directory.rglob('*') if p.is_dir()]
            for subdirectory in subdirectories:
                inferred_lesson_number = infer_lesson_number(subdirectory, infer_from="directory")
                logger.info(f"Inferred lesson number {inferred_lesson_number} from {subdirectory}")
                if lesson_range is None or inferred_lesson_number in lesson_range:
                    all_documents.extend(load_documents(subdirectory, inferred_lesson_number))
                    # Check if any subdirectory has its own subdirectory
                else:
                    logger.info(f"Skipped subdirectory {subdirectory}")
                    continue
                sub_subdirectories = [p for p in subdirectory.glob('*/') if p.is_dir()]
                if sub_subdirectories:
                    logger.warning(
                        f"Overly nested lesson directories found: {subdirectory} contains subdirectories. Readings in these directories not loaded")
        else:
            # Iterate through files in the directory and infer the lesson number from each file's name
            for file in directory.glob('*'):
                if file.is_file():
                    inferred_lesson_number = infer_lesson_number(file, infer_from)
                    logger.info(f"Inferred lesson number {inferred_lesson_number} from {file.name}")
                    if inferred_lesson_number is not None and (lesson_range is None or inferred_lesson_number in lesson_range):
                        all_documents.extend(load_documents(directory, inferred_lesson_number))
                    else:
                        logger.info(f"Skipped file {file.name} as it did not match the lesson range")

    return all_documents


def infer_lesson_number(path: Path, infer_from: str) -> int:
    """
    Infers the lesson number from either the directory name or the filename.

    Args:
        path (Path): The directory or file path.
        infer_from (str): Method to infer lesson numbers; options are "filename" or "directory".

    Returns:
        int: The inferred lesson number.
    """
    if infer_from == "filename":
        return infer_lesson_from_filename(path.name)
    elif infer_from == "directory":
        # Assuming directory name contains the lesson number
        match = re.search(r'\d+', path.name)
        if match:
            return int(match.group(0))
    return None


def infer_lesson_from_filename(filename: str) -> int:
    """Match on [Ll]esson, [Ww]eek, [Ll]ecture, and "L" followed by a number"""
    # match = re.search(r'(?:Lesson|L)?(\d+)\.\d+|Lesson\s*(\d+)|L\s*(\d+)', filename, re.IGNORECASE)
    match = re.search(
        r'(?:Lesson|L|Week|W|Lecture|Lect)?\s*(\d+)\.\d+|Lesson\s*(\d+)|L\s*(\d+)|Week\s*(\d+)|W\s*(\d+)|Lecture\s*(\d+)|Lect\s*(\d+)',
        filename,
        re.IGNORECASE
    )

    if match:
        # Return the first group that is not None
        for group in match.groups():
            if group:
                return int(group)
    return None


def clean_ocr_text(ocr_text: str) -> str:
    """
    Clean the OCR text by removing any characters that are not letters, digits, common punctuation,
    parentheses, percent signs, or whitespace.

    Args:
        ocr_text (str): The raw OCR text.

    Returns:
        str: Cleaned text with only valid characters.
    """
    # Remove unwanted characters (keep letters, digits, common punctuation, parentheses, percent signs, and spaces)
    cleaned_text = re.sub(r"[^a-zA-Z0-9.,!?;:%()\-\n\s]", "", ocr_text)

    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extracts text from a PDF file, and if needed, performs OCR to extract text from image-based PDFs.

    Args:
        pdf_path (Path): The path to the PDF file.
        use_ocr (bool): If True, apply OCR if no text is found.

    Returns:
        str: The text content of the PDF as a single string.
    """
    logger = logger_setup(log_level=logging.WARNING)
    pdf_path = Path(pdf_path)
    text_content = []

    # Attempt to extract text directly from the PDF using pypdf
    with open(str(pdf_path), 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Split the text into paragraphs (double newlines as paragraph breaks)
                paragraphs = page_text.split('\n\n')
                text_content.extend(paragraph.strip() for paragraph in paragraphs if paragraph.strip())

    # If no text is found and use_ocr is True, attempt OCR
    if not text_content:
        logger.warning(
            "No readable text found in pdf. Attempting OCR. If results are poor, recommend converting to readable text and then running again.")
        ocr_result = ocr_pdf(pdf_path, max_workers=6)
        ocr_result_clean = clean_ocr_text(ocr_result)
        return ocr_result_clean if ocr_result_clean.strip() else "OCR did not produce readable text."

    # Join and return the extracted text
    combined_text = ' '.join(text_content)
    return combined_text if combined_text.strip() else "No readable text found."


# Load readings from either a PDF or a TXT file
def load_readings(file_path: Union[str, Path]) -> str:
    """
    Loads text from a PDF, DOCX, or TXT file and returns it as a string.
    The text is prefixed with the title derived from the file name.

    Args:
        file_path (Path): The path to the file.

    Returns:
        str: The text extracted from the file.

    Raises:
        ValueError: If no readable text could be extracted from the file.
    """
    file_path = Path(file_path)

    def check_extracted_text(extracted_text: str, file_name: str):
        if not extracted_text.strip():
            raise ValueError(f"No readable text found in {file_name}. Ensure the file has content.")

    text = 'title: ' + file_path.stem + "\n"

    if file_path.suffix.lower() == '.pdf':
        extracted_text = extract_text_from_pdf(file_path)
        check_extracted_text(extracted_text, file_path.name)

    elif file_path.suffix.lower() == '.docx':
        try:
            doc = Document(str(file_path))
            extracted_text = "\n".join([para.text for para in doc.paragraphs])
        except PackageNotFoundError:
            raise ValueError(f"Unable to open {file_path.name}. The file might be corrupted or not a valid DOCX document.")
        check_extracted_text(extracted_text, file_path.name)

    elif file_path.suffix.lower() == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                extracted_text = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                extracted_text = file.read()
        check_extracted_text(extracted_text, file_path.name)

    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    return text + extracted_text


######################### Syllabus functions ###################################

def find_docx_indices(syllabus: List[str], current_lesson: int, lesson_identifier: str = None) -> Tuple[int, int, int, int]:
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


def load_docx_syllabus(syllabus_path: Union[str, Path]) -> List[str]:
    """
    Loads a DOCX syllabus and returns its content as a list of paragraphs.

    Args:
        syllabus_path (Path): The path to the DOCX file.

    Returns:
        List[str]: The syllabus content as a list of paragraphs.
    """
    syllabus_path = Path(syllabus_path)

    max_retries = 3
    retry_delay = 10  # seconds

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


def convert_pdf_to_docx(pdf_path: Union[str, Path]) -> None:
    """
    Convert a PDF file to a DOCX file.

    Args:
        pdf_path (Path): Path to the input PDF file.
        docx_path (Path): Path where the output DOCX file will be saved.
    """
    pdf_path = Path(pdf_path)
    docx_path = pdf_path.parent / f"{pdf_path.stem}.docx"

    cv = Converter(str(pdf_path))
    cv.convert(str(docx_path), start=0, end=None)  # You can specify page range if needed
    cv.close()
    print(f"Converted {pdf_path.name} to {docx_path.name}")

    return docx_path


def extract_lesson_objectives(syllabus_path: Union[str, Path], current_lesson: int, only_current: bool = False) -> str:
    """
    Extracts objectives for the previous, current, and next lessons from the syllabus.

    Args:
        syllabus_path (Path): The path to the syllabus document (PDF or DOCX).
        current_lesson (int): The current lesson number.

    Returns:
        str: The objectives for the previous, current, and next lessons.
    """
    syllabus_path = Path(syllabus_path)
    if syllabus_path.suffix not in [".docx", ".pdf"]:
        raise ValueError(f"Unsupported file type: {syllabus_path.suffix}")

    if syllabus_path.suffix == '.pdf':
        old_syllabus_path = syllabus_path
        syllabus_path = convert_pdf_to_docx(old_syllabus_path)

    syllabus_content = load_docx_syllabus(syllabus_path)
    prev_idx, curr_idx, next_idx, end_idx = find_docx_indices(syllabus_content, current_lesson)
    prev_lesson_content = "\n".join(syllabus_content[prev_idx:curr_idx]) if prev_idx is not None else ""
    curr_lesson_content = "\n".join(syllabus_content[curr_idx:next_idx]) if curr_idx is not None else ""
    next_lesson_content = "\n".join(syllabus_content[next_idx:end_idx]) if next_idx is not None else ""

    combined_content = "\n".join(filter(None, [prev_lesson_content, curr_lesson_content, next_lesson_content]))

    if only_current:
        return curr_lesson_content

    return combined_content


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv
    load_dotenv()

    # Path definitions
    syllabus_path = Path(os.getenv('syllabus_path'))
    pdf_syllabus_path = Path(os.getenv('pdf_syllabus_path'))
    readingsDir = Path(os.getenv('readingsDir'))

    lsn = 8
    lsn_objectives_doc = extract_lesson_objectives(syllabus_path,
                                                   lsn,
                                                   only_current=True)
    converted_syllabus_path = convert_pdf_to_docx(pdf_syllabus_path)
    converted_objectives = extract_lesson_objectives(converted_syllabus_path,
                                                     lsn,
                                                     only_current=True)
    lsn_objectives_pdf = extract_lesson_objectives(pdf_syllabus_path,
                                                   lsn,
                                                   only_current=True)

    print(f"doc objectives:\n{lsn_objectives_doc}")
    print(f"\npdf objectives:\n{lsn_objectives_pdf}")

    docs = load_lessons(readingsDir, recursive=True, lesson_range=range(20, 21))

    ocr_test = Path(readingsDir / "L21/21.3 Pew Research Center. Beyond Red vs Blue Overview_ocr.pdf")
    ocr_result = extract_text_from_pdf(ocr_test)
    print(ocr_result)
