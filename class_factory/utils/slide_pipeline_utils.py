import re
import subprocess
import tempfile
from pathlib import Path


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


def validate_latex(latex_code: str, latex_compiler: str = "pdflatex") -> bool:
    """
    Validates LaTeX by attempting to compile it using a LaTeX engine.

    Args:
        latex_code (str): The LaTeX code to validate.
        latex_compiler (str): The full path or name of the LaTeX compiler executable if it's not on the PATH.

    Returns:
        bool: True if LaTeX compiles successfully, False otherwise.
    """
    compiler_cmd = Path(latex_compiler)

    # Ensure we use the path or just the name if it’s in PATH
    compiler_cmd_str = str(compiler_cmd) if compiler_cmd.is_file() else latex_compiler

    # Create a temporary directory to save the LaTeX file and compile it
    with tempfile.TemporaryDirectory() as tempdir:
        tex_file = Path(tempdir) / "tempfile.tex"

        # Write the LaTeX code to the temporary .tex file
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(latex_code)

        # Try to compile the LaTeX file using pdflatex
        try:
            result = subprocess.run(
                [compiler_cmd_str, "-interaction=nonstopmode", str(tex_file)],
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
                return False

        except subprocess.TimeoutExpired:
            print("LaTeX compilation timed out!")
            return False
        except Exception as e:
            print(f"LaTeX compilation failed with error {e}")
            return False


# def load_beamer_presentation(tex_path: Path) -> str:
#     """
#     Loas a Beamer presentation from a .tex file and returns it as a string.

#     Args:
#         tex_path (Path): The path to the .tex file containing the Beamer presentation.
#     Returns:
#         str: The content of the .tex file.
#     """
#     with open(tex_path, 'r', encoding='utf-8') as file:
#         beamer_text = file.read()
#     return beamer_text


# clean response
def clean_latex_content(latex_content: str) -> str:
    """
    Clean LaTeX content by removing any text before the \title command and
    stripping extraneous LaTeX code blocks markers.

    Args:
        latex_content (str): The LaTeX content to be cleaned.

    Returns:
        str: The cleaned LaTeX content.
    """
    # Find the position of the \title command
    title_position = latex_content.find(r'\title')

    if title_position != -1:
        # Keep only the content starting from \title
        cleaned_content = latex_content[title_position:]
    else:
        # If \title is not found, return the original content (or handle as needed)
        cleaned_content = latex_content

    # Remove any ```latex or ``` markers at the beginning and end
    if cleaned_content.startswith("```latex"):
        cleaned_content = cleaned_content[len("```latex"):].lstrip()
    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content[len("```"):].lstrip()

    cleaned_content = cleaned_content.rstrip("```").rstrip()

    # Escape standalone dollar signs not in math mode
    # Matches dollar signs not within $...$ or $$...$$
    cleaned_content = re.sub(
        r'(?<!\$)(?<!\\)\$(?![^\$]*\$)',
        r'\$',
        cleaned_content
    )

    # Remove ampersands, as long as they're not part of a tabular environment
    if r'\begin{tabular}' not in cleaned_content:
        # Anything not in a tabular environment will be escaped with \
        cleaned_content = re.sub(r'\\&', 'and', cleaned_content)

    return cleaned_content
