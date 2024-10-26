import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pyprojroot.here import here

from class_factory.utils.load_documents import (extract_lesson_objectives,
                                                extract_text_from_pdf,
                                                infer_lesson_number,
                                                load_documents,
                                                load_docx_syllabus,
                                                load_lessons, load_readings)

wd = here()


@pytest.fixture
def mock_files():
    # Create mock files with different lesson numbers and types
    mock_txt = MagicMock(spec=Path)
    mock_txt.suffix = '.txt'
    mock_txt.name = 'Lesson1.txt'
    mock_txt.read_text.return_value = "Lesson 1 TXT content"

    mock_pdf = MagicMock(spec=Path)
    mock_pdf.suffix = '.pdf'
    mock_pdf.name = 'L2.pdf'

    mock_docx = MagicMock(spec=Path)
    mock_docx.suffix = '.docx'
    mock_docx.name = 'Lesson 3.docx'

    # Mock a directory structure
    mock_directory = MagicMock(spec=Path)
    mock_directory.glob.return_value = [mock_txt, mock_pdf, mock_docx]

    return [mock_txt, mock_pdf, mock_docx], mock_directory


class MockPath:
    def __init__(self, path, is_file=True, children=None):
        self.path = path
        self._is_file = is_file
        self.name = Path(path).name
        self.stem = Path(path).stem
        self.suffix = Path(path).suffix
        self._children = children or []

    def __str__(self):
        return self.path

    def __fspath__(self):
        return self.path

    def is_file(self):
        return self._is_file

    def is_dir(self):
        return not self._is_file

    def glob(self, pattern):
        if not self.is_dir():
            return []
        return self._children

    def rglob(self, pattern):
        if not self.is_dir():
            return []
        result = self._children[:]
        for child in self._children:
            if child.is_dir():
                result.extend(child.rglob(pattern))
        return result


@pytest.fixture
def mock_recursive_files():
    # Create mock files
    mock_txt_main = MockPath('mock_dir/Lesson1.txt')
    mock_txt_sub = MockPath('mock_dir/week2/Lesson2.txt')
    # Create a mock subdirectory containing the sub file
    mock_subdir = MockPath('mock_dir/week2', is_file=False, children=[mock_txt_sub])
    # Create a mock main directory containing the main file and the subdirectory
    mock_directory = MockPath('mock_dir', is_file=False, children=[mock_txt_main, mock_subdir])

    return mock_directory, [mock_txt_main, mock_txt_sub]


def test_load_lessons(mock_files):
    mock_files, mock_directory = mock_files

    with patch('class_factory.utils.load_documents.Path.glob', return_value=mock_files):
        # Mock the load_readings function to return sample content based on file type
        with patch('class_factory.utils.load_documents.load_readings') as mock_load_readings:
            mock_load_readings.side_effect = [
                "title: Lesson1\nLesson 1 TXT content",
                "title: Lesson2\nLesson 2 PDF content",
                "title: Lesson3\nLesson 3 DOCX content"
            ]

            # Test loading lessons from the mock directory
            lessons = load_lessons(mock_directory, infer_from='filename', recursive=False)
            assert len(lessons) == 3
            assert "Lesson 1 TXT content" in lessons[0]
            assert "Lesson 2 PDF content" in lessons[1]
            assert "Lesson 3 DOCX content" in lessons[2]


def test_load_lessons_recursive(mock_recursive_files):
    mock_directory, mock_files = mock_recursive_files

    # Correct the module path to match where 'infer_lesson_number' and 'load_documents' are imported in 'load_lessons'
    with patch('class_factory.utils.load_documents.infer_lesson_number') as mock_infer_lesson_number:
        mock_infer_lesson_number.side_effect = lambda path, infer_from: int(
            re.findall(r'\d+', path.name)[0]) if re.findall(r'\d+', path.name) else None

        with patch('class_factory.utils.load_documents.load_documents') as mock_load_documents:
            def mock_load_documents_func(directory, lesson_number):
                return [f"title: Lesson{lesson_number}\nLesson {lesson_number} content"]
            mock_load_documents.side_effect = mock_load_documents_func

            # Run the load_lessons function with recursive=False
            lessons = load_lessons([mock_directory], infer_from='filename', recursive=False)
            assert len(lessons) == 1
            assert "Lesson 1 content" in lessons[0]

            # Run the load_lessons function with recursive=True
            lessons = load_lessons([mock_directory], infer_from='filename', recursive=True)
            assert len(lessons) == 1
            assert "Lesson 2 content" in lessons[0]


def test_load_documents(mock_files):
    mock_files, mock_directory = mock_files

    with patch('class_factory.utils.load_documents.Path.glob', return_value=mock_files):
        # Mock the extract_text_from_pdf and load_readings functions to simulate content extraction
        with patch('class_factory.utils.load_documents.extract_text_from_pdf', return_value="Lesson 2 PDF content"):
            with patch('class_factory.utils.load_documents.load_readings', side_effect=[
                "Lesson 1 TXT content",
                "Lesson 2 PDF content",
                "Lesson 3 DOCX content"
            ]):
                # Test loading documents with lesson number inference
                documents = load_documents(mock_directory, lesson_number=1)
                assert len(documents) == 1
                assert "Lesson 1 TXT content" in documents[0]

                documents = load_documents(mock_directory, lesson_number=2)
                assert len(documents) == 1
                assert "Lesson 2 PDF content" in documents[0]

                documents = load_documents(mock_directory, lesson_number=3)
                assert len(documents) == 1
                assert "Lesson 3 DOCX content" in documents[0]


def test_infer_lesson_number():
    # Test inferring from filename
    assert infer_lesson_number(Path("Lesson1.pdf"), infer_from="filename") == 1
    assert infer_lesson_number(Path("Week 2.txt"), infer_from="filename") == 2
    assert infer_lesson_number(Path("Lecture3.docx"), infer_from="filename") == 3
    assert infer_lesson_number(Path("NoLesson.docx"), infer_from="filename") is None

    # Test inferring from directory name
    assert infer_lesson_number(Path("/path/to/Lesson1"), infer_from="directory") == 1
    assert infer_lesson_number(Path("/path/to/Week2"), infer_from="directory") == 2
    assert infer_lesson_number(Path("/path/to/Lecture 3"), infer_from="directory") == 3
    assert infer_lesson_number(Path("/path/to/AllLessons"), infer_from="directory") is None


def test_extract_text_from_pdf():
    # Mock a valid PDF file scenario
    mock_pdf_path = MagicMock(spec=Path)
    mock_pdf_path.suffix = '.pdf'
    mock_pdf_path.name = 'Lesson2.pdf'
    mock_pdf_path.__str__.return_value = 'mock_path/Lesson2.pdf'  # Add this line

    # Mock the pypdf PdfReader behavior
    with patch('class_factory.utils.load_documents.pypdf.PdfReader') as MockPdfReader, \
            patch('builtins.open', mock_open()) as mock_file:  # Add mock_open
        # Set up the mock PdfReader
        mock_reader = MockPdfReader.return_value
        mock_reader.pages = [MagicMock()]
        mock_reader.pages[0].extract_text.return_value = "Lesson 2 PDF content"

        # Test extracting text from the mocked PDF
        text = extract_text_from_pdf(mock_pdf_path)
        assert text == "Lesson 2 PDF content"

    # Mock a corrupted PDF file scenario
    with patch('class_factory.utils.load_documents.pypdf.PdfReader', side_effect=ValueError("Invalid PDF")), \
            patch('builtins.open', mock_open()) as mock_file:  # Add mock_open
        with pytest.raises(ValueError):
            extract_text_from_pdf(mock_pdf_path)


def test_load_readings():
    # Test for TXT file
    mock_txt_path = MockPath('lesson3.txt')
    txt_content = "Lesson 3 TXT content"
    with patch('builtins.open', mock_open(read_data=txt_content)):
        readings = load_readings(mock_txt_path)
        assert "title: lesson3\n" + txt_content in readings

    # Test for PDF file
    mock_pdf_path = MockPath('lesson3.pdf')
    pdf_content = "Lesson 3 PDF content"
    with patch('class_factory.utils.load_documents.extract_text_from_pdf', return_value=pdf_content):
        readings = load_readings(mock_pdf_path)
        assert "title: lesson3\n" + pdf_content in readings

    # Test for DOCX file
    mock_docx_path = MockPath('lesson3.docx')
    docx_content = "Lesson 3 DOCX content"
    mock_document = MagicMock()
    mock_document.paragraphs = [MagicMock(text=docx_content)]
    with patch('class_factory.utils.load_documents.Document', return_value=mock_document):
        readings = load_readings(mock_docx_path)
        assert "title: lesson3\n" + docx_content in readings

    # Test for unsupported file type (MD)
    mock_md_path = MockPath('lesson3.md')
    with pytest.raises(ValueError, match=r"Unsupported file type: \.md"):
        load_readings(mock_md_path)

    # Test for file not found
    mock_not_found_path = MockPath('not_found.txt')
    mock_not_found_path._is_file = False
    with pytest.raises(FileNotFoundError, match=r"No such file or directory: "):
        load_readings(mock_not_found_path)

    # Test for empty file
    mock_empty_path = MockPath('empty.txt')
    with patch('builtins.open', mock_open(read_data="")):
        with pytest.raises(ValueError, match=r"No readable text found in empty\.txt"):
            load_readings(mock_empty_path)


def test_load_docx_syllabus():
    # Mock the syllabus file path
    mock_syllabus_path = MagicMock(spec=Path)
    mock_syllabus_path.suffix = '.docx'
    mock_syllabus_path.is_file.return_value = True

    # Mock Document to simulate loading paragraphs
    mock_document = MagicMock()
    mock_paragraph_1 = MagicMock()
    mock_paragraph_2 = MagicMock()
    mock_paragraph_1.text = "Lesson 1: Introduction"
    mock_paragraph_2.text = "Lesson 2: Advanced Topics"

    # Mock the paragraphs attribute
    mock_document.paragraphs = [mock_paragraph_1, mock_paragraph_2]

    with patch('class_factory.utils.load_documents.Document', return_value=mock_document):
        syllabus_content = load_docx_syllabus(mock_syllabus_path)
        assert len(syllabus_content) == 2
        assert "Lesson 1: Introduction" in syllabus_content
        assert "Lesson 2: Advanced Topics" in syllabus_content


def test_extract_lesson_objectives():
    # Create a MockPath instance
    mock_syllabus_path = MockPath('mock_syllabus.docx')

    # Mock the syllabus content
    mock_syllabus_content = [
        "Lesson 1: Introduction",
        "Lesson 2: Advanced Topics",
        "Lesson 3: Summary"
    ]

    # Use multiple patch decorators
    @patch('class_factory.utils.load_documents.load_docx_syllabus', return_value=mock_syllabus_content)
    @patch('class_factory.utils.load_documents.find_docx_indices', return_value=(0, 1, 2, 3))
    def run_test(mock_find_indices, mock_load_syllabus):
        objectives = extract_lesson_objectives(mock_syllabus_path, current_lesson=2)
        assert "Lesson 2: Advanced Topics" in objectives
        assert "Lesson 3: Summary" in objectives

    run_test()

    # Test for unsupported file type
    mock_unsupported_path = MockPath('mock_syllabus.txt')
    with pytest.raises(ValueError, match=r"Unsupported file type: \.txt"):
        extract_lesson_objectives(mock_unsupported_path, current_lesson=1)


if __name__ == "__main__":
    pytest.main([__file__])
