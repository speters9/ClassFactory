from pathlib import Path


def load_beamer_presentation(tex_path: Path) -> str:
    """
    Loas a Beamer presentation from a .tex file and returns it as a string.

    Args:
        tex_path (Path): The path to the .tex file containing the Beamer presentation.
    Returns:
        str: The content of the .tex file.
    """
    with open(tex_path, 'r', encoding='utf-8') as file:
        beamer_text = file.read()
    return beamer_text


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

    cleaned_content = cleaned_content.lstrip("```latex\n").rstrip("```")
    return cleaned_content
