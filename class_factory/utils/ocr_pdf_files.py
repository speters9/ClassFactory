"""
OCR PDF/Text Extraction Utilities

Sections:
1. Docling-based PDF OCR (for digital/text PDFs)
2. Image-based PDF OCR (for scanned/image PDFs)
3. Unified Pipeline (easy import for main use)
"""
#%%
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pytesseract
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (AcceleratorDevice,
                                                AcceleratorOptions,
                                                PdfPipelineOptions,
                                                TesseractCliOcrOptions)

from docling.document_converter import DocumentConverter, PdfFormatOption
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
from textblob import TextBlob
from tqdm import tqdm

user_home = Path.home()

pytesseract.pytesseract.tesseract_cmd = str(user_home / r'AppData\Local\Programs\Tesseract-OCR\tesseract.exe')

# =============================
# 1. DOCLING-BASED PDF OCR
# =============================
#%%

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
# Point to tesseract executable

# %%


def ocr_pdf_docling(pdf_path: Path, max_workers: int = 8) -> str:
    """
    Extract text from a PDF using the Docling pipeline.
    Args:
        pdf_path (Path): Path to the PDF file.
        max_workers (int): Number of threads to use for parallel processing.
    Returns:
        str: Cleaned OCR result as a string.
    """
    str_path = str(pdf_path)
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

# =============================
# 2. IMAGE-BASED PDF OCR (SCANNED)
# =============================





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
    doc = TextBlob(strings)
    cleaned = doc.correct().string
    return cleaned


def ocr_image_tesseract(image_path: Path, contrast: float = 1.2, sharpen: bool = True, replace_dict: dict = None) -> str:
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
        img = img.convert('L')
        img = preprocess_background_to_white(img, threshold=240)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
        if sharpen:
            img = img.filter(ImageFilter.SHARPEN)
        if table:
            img.save(temp_path)
            ocr = TesseractOCR(n_threads=1, psm=6, lang="eng")
            doc = Img2TableImage(temp_path)
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
            text = pytesseract.image_to_string(img, config='--psm 3 --oem 1 --dpi 400')
            doc = TextBlob(text)
            cleaned = doc.correct().string
            if replace_dict:
                for k, v in replace_dict.items():
                    cleaned = cleaned.replace(k, v)
            return cleaned
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return ""
    finally:
        if Path(temp_path).exists():
            os.remove(temp_path)


def process_pdf_page(image, page_number: int):
    """
    Process a single page of the PDF, save it as a temporary image file, perform OCR, and clean up.
    """
    temp_image_path = Path(f"temp_pdf_page_{page_number}.png")
    image.save(temp_image_path)

    # Perform OCR on the image
    ocr_text = ocr_image_tesseract(temp_image_path)

    # Ensure the temp file is removed after processing
    if Path(temp_image_path).exists():
        os.remove(temp_image_path)

    # Return the OCR result
    return page_number, ocr_text

# =============================
# 3. UNIFIED PIPELINE (MAIN ENTRY POINT)
# =============================


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



def ocr_pdf(pdf_path: Path, max_workers: int = 4, pipeline: str = 'docling') -> str:
    """
    Unified PDF-to-text pipeline. Use pipeline='docling' for Docling, pipeline='image' for image-based OCR.
    Args:
        pdf_path (Path): Path to the PDF file.
        max_workers (int): Number of threads to use for parallel processing.
        pipeline (str): 'docling' or 'image'.
    Returns:
        str: Cleaned OCR result as a string.
    """
    if pipeline == 'docling':
        return ocr_pdf_docling(pdf_path, max_workers=max_workers)
    elif pipeline == 'image':
        # Image-based fallback: convert PDF to images, OCR each page
        images = convert_from_path(
            str(pdf_path),
            dpi=400,
            fmt='png',
            thread_count=4,
        )
        text_content = ['']*len(images)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_pdf_page, image, page_number)
                       for page_number, image in enumerate(images)]
            for future in tqdm(as_completed(futures), total=len(futures)):
                page_number, result = future.result()
                text_content[page_number] = result
        ocr_result = ' '.join(text_content)
        return clean_ocr_text(ocr_result)
    else:
        raise ValueError("pipeline must be 'docling' or 'image'")


#%% 

if __name__ == "__main__":
    # Example usage for one image
    # img_path = Path("C:/Users/Sean/OneDrive - afacademy.af.edu/Documents/Classes/Fall 2024/PS211/02_Class Readings/L11/snips/p141_house_senate_diff_table.png")
    # result = ocr_image(img_path, replace_dict = {'Indian': 'American'})

    from dotenv import load_dotenv
    load_dotenv()

    readingsDir = Path("C:/Users/Sean/OneDrive - afacademy.af.edu/Documents/Classes/Fall 2024/PS211/02_Class Readings/L21/reference")
    pdf_path = readingsDir / "21.3 Pew Research Center. Beyond Red vs Blue Overview.pdf"
    ocr_result = ocr_pdf(pdf_path, max_workers=6, pipeline='docling')
    print(ocr_result)


# %%
