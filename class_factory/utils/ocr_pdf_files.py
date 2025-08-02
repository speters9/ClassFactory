"""
Convert image data to text for inclusion in beamerbot pipeline
"""
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
from img2table.ocr import TesseractOCR
from img2table.document import Image as Img2TableImage
import spacy
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import os
import pytesseract
import re
from pathlib import Path
from typing import List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (AcceleratorDevice,
                                                AcceleratorOptions,
                                                EasyOcrOptions, OcrMacOptions,
                                                PdfPipelineOptions,
                                                RapidOcrOptions,
                                                TesseractCliOcrOptions,
                                                TesseractOcrOptions)
# %%
from docling.document_converter import DocumentConverter, PdfFormatOption
from textblob import TextBlob

user_home = Path.home()

pytesseract.pytesseract.tesseract_cmd = str(user_home / r'AppData\Local\Programs\Tesseract-OCR\tesseract.exe')
# %%

# import contextualSpellCheck

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
# Point to tesseract executable

# Initialize spacy and add contextual spell checker
# nlp = spacy.load('en_core_web_lg')
# contextualSpellCheck.add_to_pipe(nlp)
nlp = SpellChecker()

# %%


def ocr_image(image_path: Path | str, max_workers: int = 8) -> List[str]:
    """
    Perform OCR on a list of image files in parallel.

    Args:
        image_paths (List[Path]): List of paths to the image files to be processed.
        max_workers (int): Number of threads to use for parallel processing.

    Returns:
        List[str]: List of OCR results as strings.
    """
    str_path = str(image_path)

    accelerator_options = AcceleratorOptions(
        num_threads=max_workers, device=AcceleratorDevice.AUTO
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    doc = converter.convert(str_path)
    text = doc.document.export_to_markdown()

    cleaned_text = clean_ocr_text(text)

    return cleaned_text


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
    cleaned_text = str(TextBlob(ocr_text).correct())

    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def preprocess_background_to_white(img: Image.Image, threshold: int = 235) -> Image.Image:
    """
    Convert image background to white by thresholding light colors.

    Args:
        img (PIL.Image.Image): Input image to be processed.
        threshold (int): Threshold value (0-255) above which all pixels will be set to white.

    Returns:
        PIL.Image.Image: Processed image with background turned to white.
    """
    # Convert image to numpy array
    img_np = np.array(img)

    # Check if image is grayscale, otherwise convert to grayscale
    if img_np.ndim == 3:  # If it's an RGB image
        img_gray = np.mean(img_np, axis=2)  # Convert to grayscale by averaging the RGB channels
    else:
        img_gray = img_np

    # Create a mask for pixels that are lighter than the threshold
    mask = img_gray > threshold

    # Set those pixels to pure white (255)
    img_np[mask] = 255

    # Convert back to a PIL image
    img_white_bg = Image.fromarray(img_np)

    return img_white_bg


def process_extracted_table(df):
    """
    Process the extracted table by converting to string and applying spell checking.

    Args:
        df (pd.DataFrame): DataFrame of the extracted table.

    Returns:
        str: Cleaned and spell-checked string.
    """
    strings = df.to_string()
    doc = nlp(strings)
    cleaned = doc._.outcome_spellCheck
    return cleaned


def ocr_image(image_path: Path, contrast: float = 1.2, sharpen: bool = True, replace_dict: dict = None) -> str:
    """
    Perform OCR on an image file, enhance it, and correct spelling using a contextual spell checker.

    Args:
        image_path (Path): Path to the image file to be processed.
        contrast (float): Factor by which to increase contrast (default 2.0).
        sharpen (bool): Whether to apply sharpening to the image (default True).

    Returns:
        str: The corrected text extracted from the image.
    """
    table = 'table' in image_path.name  # False if not a table
    temp_path = "temp_preprocessed_image.png"  # Use Path object for temp file

    try:
        img = Image.open(image_path)

        # Convert to grayscale and clean the background
        img = img.convert('L')
        img = preprocess_background_to_white(img, threshold=240)

        # Increase contrast and sharpen if needed
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
        if sharpen:
            img = img.filter(ImageFilter.SHARPEN)

        if table:
            img.save(temp_path)

            # Initialize TesseractOCR for tables
            ocr = TesseractOCR(n_threads=1, psm=6, lang="eng")
            doc = Img2TableImage(temp_path)

            # Try extracting tables with explicit borders
            extracted_tables = doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                implicit_columns=True,
                borderless_tables=False,
                min_confidence=45
            )

            if extracted_tables:
                cleaned = process_extracted_table(extracted_tables[0].df)
                return cleaned

            # Retry without explicit table borders
            extracted_tables = doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                implicit_columns=True,
                borderless_tables=True,
                min_confidence=45
            )

            if extracted_tables:
                cleaned = process_extracted_table(extracted_tables[0].df)
                return cleaned

        else:
            # Perform OCR for non-table images
            text = pytesseract.image_to_string(img, config='--psm 3 --oem 1 --dpi 400')

            doc = nlp(text)
            cleaned = doc._.outcome_spellCheck

            if replace_dict:
                for k, v in replace_dict.items():
                    cleaned = cleaned.replace(k, v)

            return cleaned

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return ""

    finally:
        # Ensure the temp file is removed after processing
        if Path(temp_path).exists():
            os.remove(temp_path)


def process_pdf_page(image, page_number: int):
    """
    Process a single page of the PDF, save it as a temporary image file, perform OCR, and clean up.
    """
    temp_image_path = Path(f"temp_pdf_page_{page_number}.png")
    image.save(temp_image_path)

    # Perform OCR on the image
    ocr_text = ocr_image(temp_image_path)

    # Ensure the temp file is removed after processing
    if Path(temp_image_path).exists():
        os.remove(temp_image_path)

    # Return the OCR result
    return page_number, ocr_text


def ocr_pdf(pdf_path: Path, max_workers: int = 4):
    """
    Convert PDF to images and perform OCR on each page in parallel.

    Args:
        pdf_path (Path): Path to the PDF file.
        max_workers (int): Number of threads to use for parallel processing.

    Returns:
        str: Full OCR result as a string.
    """
    images = convert_from_path(
        str(pdf_path),
        dpi=400,
        fmt='png',
        thread_count=4,  # This sets the number of threads for PDF to image conversion
    )

    text_content = ['']*len(images)  # empty list to receive indexed futures

    # Use ThreadPoolExecutor to process PDF pages in parallel. Results are indexed by image order
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pdf_page, image, page_number)
                   for page_number, image in enumerate(images)]  # set defined page number for page to insert in order

        for future in tqdm(as_completed(futures), total=len(futures)):
            page_number, result = future.result()  # Collect both page_number and OCR result
            text_content[page_number] = result  # Store the result in the correct position

    # Combine the OCR results from all pages into a single string
    ocr_result = ' '.join(text_content)
    return ocr_result


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


if __name__ == "__main__":
    # Example usage for one image
    # img_path = Path("C:/Users/Sean/OneDrive - afacademy.af.edu/Documents/Classes/Fall 2024/PS211/02_Class Readings/L11/snips/p141_house_senate_diff_table.png")
    # result = ocr_image(img_path, replace_dict = {'Indian': 'American'})

    from dotenv import load_dotenv
    load_dotenv()

    readingsDir = Path("C:/Users/Sean/OneDrive - afacademy.af.edu/Documents/Classes/Fall 2024/PS211/02_Class Readings/L21/reference")
    pdf_path = readingsDir / "21.3 Pew Research Center. Beyond Red vs Blue Overview.pdf"
    ocr_result = ocr_pdf(pdf_path, max_workers=6)
    print(ocr_result)
