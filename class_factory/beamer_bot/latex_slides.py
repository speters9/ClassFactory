"""
latex_slides.py
---------------

Defines structured classes for robust LaTeX Beamer slide generation.
"""
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from class_factory.beamer_bot.latex_utils import clean_latex_content

LATEX_SPECIAL_CHARS = {
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    # NOTE: Do NOT escape braces in LaTeX content! Only escape in plain-text fields like titles.
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
    '\\': r'\textbackslash{}',
}


def escape_latex(text: str) -> str:
    """
    Escape LaTeX special characters in a string.
    WARNING: Only use this for plain-text fields (e.g., slide titles).
    Do NOT use on LaTeX content, as it will break valid LaTeX code.
    """
    return re.sub(r'([&%$#_{}~^\\])', lambda m: LATEX_SPECIAL_CHARS[m.group()], text)


class LatexSlide(BaseModel):
    title: str = Field(description="The title of the slide")
    content: str = Field(description="The LaTeX content of the slide")
    slide_type: Optional[str] = Field(description="The type of slide, e.g., 'title', 'objectives', 'titlepage', etc. This will determine the slide's position in the deck. \
                                      The title slide must have slide_type 'titlepage' to be rendered correctly.")

    def to_latex(self) -> str:
        # Special handling for titlepage slide
        if self.slide_type == "titlepage":
            return "\\begin{frame}\n\\titlepage\n\\end{frame}"

        title = escape_latex(self.title)
        clean_content = clean_latex_content(self.content)
        # Content is assumed to be valid LaTeX, do NOT escape it.
        return f"\\begin{{frame}}{{{title}}}\n{clean_content}\n\\end{{frame}}"


class LatexSlides(BaseModel):
    slides: List[LatexSlide] = Field(description="A list of LatexSlide objects representing the presentation slides")
    title: Optional[str] = Field(
        default=None, description="Title of this lesson. The title of this lesson number should be included in the lesson objectives.")
    author: str = Field(default="", description="Author name for the presentation (you can pull this from the prior presentation)")
    institute: str = Field(default="", description="Institute name for the presentation (you can pull this from the prior presentation)")
    date: str = Field(default="\\today", description="Date of the presentation")

    def to_latex(self) -> str:
        meta = f"\n" \
            f"\\title{{{self.title}}}\n" \
            f"\\author{{{self.author}}}\n" \
            f"\\institute{{{self.institute}}}\n" \
            f"\\date{{{self.date}}}\n\n"
        slides_latex = "\n\n".join(slide.to_latex() for slide in self.slides)
        return f"{meta}\\begin{{document}}\n\n{slides_latex}\n\\end{{document}}\n"


# Backward compatibility aliases
Slide = LatexSlide
BeamerSlides = LatexSlides

# Example LLM output schema (for prompt):
# [
#   {"title": "Lesson Objectives", "content": "\\begin{itemize} ... \\end{itemize}", "slide_type": "objectives"},
#   {"title": "Key Takeaways", "content": "...", "slide_type": "summary"},
#   ...
# ]
