import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from class_factory.utils.ocr_pdf_files import (ocr_image, ocr_pdf,
                                               preprocess_background_to_white)


@pytest.fixture
def mock_image_open():
    with patch('class_factory.utils.ocr_pdf_files.Image.open') as mock:
        # Mock an image
        img_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img = Image.fromarray(img_array)
        mock.return_value = img
        yield mock


@pytest.fixture
def mock_pytesseract():
    with patch('class_factory.utils.ocr_pdf_files.pytesseract.image_to_string') as mock:
        mock.return_value = "Test OCR output"
        yield mock


@pytest.fixture
def mock_convert_from_path():
    with patch('class_factory.utils.ocr_pdf_files.convert_from_path') as mock:
        # Mock the conversion to return a list of mock images
        mock_image = MagicMock(spec=Image.Image)
        mock.return_value = [mock_image] * 3
        yield mock


@pytest.fixture
def mock_process_pdf_page():
    with patch('class_factory.utils.ocr_pdf_files.process_pdf_page') as mock:
        # Mock the process_pdf_page to return page number and some text
        mock.side_effect = [(0, "Page 1 text"), (1, "Page 2 text"), (2, "Page 3 text")]
        yield mock


@pytest.fixture
def mock_os_remove():
    with patch('class_factory.utils.ocr_pdf_files.os.remove') as mock:
        yield mock


@pytest.fixture
def mock_spell_check():
    with patch('class_factory.utils.ocr_pdf_files.nlp') as mock_nlp:
        # Mock the spell-checking pipeline to return the input text without changes
        mock_doc = MagicMock()
        mock_doc._.outcome_spellCheck = "Test OCR output"
        mock_nlp.return_value = mock_doc
        yield mock_nlp


def test_ocr_image(mock_image_open, mock_pytesseract, mock_spell_check):
    # Call the function with a dummy path
    dummy_path = Path("dummy_image.png")
    result = ocr_image(dummy_path)

    # Debugging: Print the result for analysis
    print(f"OCR result: {result}")

    # Check that the result matches the mocked spell-checked output
    assert result == "Test OCR output"


def test_preprocess_background_to_white(mock_image_open):
    # Mock an image open
    img = mock_image_open.return_value

    # Call the function
    processed_img = preprocess_background_to_white(img, threshold=200)

    # Assert that the processed image is still an instance of PIL.Image
    assert isinstance(processed_img, Image.Image)


def test_ocr_pdf(mock_convert_from_path, mock_process_pdf_page):
    # Call the function with a dummy PDF path
    dummy_pdf_path = Path("dummy.pdf")
    result = ocr_pdf(dummy_pdf_path)

    # Check that the OCR result contains text from all pages
    assert "Page 1 text" in result
    assert "Page 2 text" in result
    assert "Page 3 text" in result


def test_ocr_image_handles_exception(mock_image_open, mock_os_remove):
    # Mock an image opening failure
    mock_image_open.side_effect = Exception("Failed to open image")

    # Call the function with a dummy path
    dummy_path = Path("dummy_image.png")
    result = ocr_image(dummy_path)

    # Check that the result is an empty string
    assert result == ""

    # Verify that os.remove was not called since no temp file was created
    mock_os_remove.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
