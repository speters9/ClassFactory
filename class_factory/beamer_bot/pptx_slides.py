"""
pptx_slides.py
--------------

Defines structured classes for robust PowerPoint slide generation.
These models are designed to work with python-pptx library for creating .pptx files.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class PptxSlide(BaseModel):
    """
    Represents a single PowerPoint slide with content and metadata.

    Unlike LaTeX slides, PowerPoint slides use plain text with markdown-like
    formatting and separate fields for different content types.
    """
    title: str = Field(description="The title of the slide")
    content: str = Field(
        default="",
        description="Main text content for the slide. Use **bold** and *italic* for emphasis. Keep concise."
    )
    bullet_points: Optional[List[str]] = Field(
        default=None,
        description="List of bullet points for the slide. Use **bold** for emphasis in bullets."
    )
    slide_type: Optional[str] = Field(
        default="content",
        description="The type of slide: 'title', 'objectives', 'content', 'summary', 'exercise', 'discussion', etc."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Speaker notes for the slide (not visible to audience)"
    )
    layout: Optional[str] = Field(
        default="title_and_content",
        description="PowerPoint layout to use: 'title_only', 'title_and_content', 'two_column', 'blank', etc."
    )

    def to_pptx_data(self) -> dict:
        """
        Convert slide to a dictionary format suitable for python-pptx processing.

        Returns:
            dict: Slide data with all fields for PowerPoint generation
        """
        return {
            "title": self.title,
            "content": self.content,
            "bullet_points": self.bullet_points or [],
            "slide_type": self.slide_type,
            "notes": self.notes or "",
            "layout": self.layout
        }


class PptxSlides(BaseModel):
    """
    Represents a complete PowerPoint presentation with metadata.

    This model contains all slides and presentation-level information
    like title, author, and institution.
    """
    slides: List[PptxSlide] = Field(
        description="A list of PptxSlide objects representing the presentation slides"
    )
    title: Optional[str] = Field(
        default=None,
        description="Title of this lesson presentation"
    )
    subtitle: Optional[str] = Field(
        default=None,
        description="Subtitle for the presentation (e.g., lesson number or date)"
    )
    author: str = Field(
        default="",
        description="Author name for the presentation"
    )
    institute: str = Field(
        default="",
        description="Institute name for the presentation"
    )
    date: Optional[str] = Field(
        default=None,
        description="Date of the presentation (if None, current date will be used)"
    )

    def to_pptx_data(self) -> dict:
        """
        Convert entire presentation to a dictionary format for python-pptx.

        Returns:
            dict: Complete presentation data including metadata and all slides
        """
        return {
            "metadata": {
                "title": self.title,
                "subtitle": self.subtitle,
                "author": self.author,
                "institute": self.institute,
                "date": self.date
            },
            "slides": [slide.to_pptx_data() for slide in self.slides]
        }

    def get_slide_count(self) -> int:
        """Get the total number of slides in the presentation."""
        return len(self.slides)

    def get_slides_by_type(self, slide_type: str) -> List[PptxSlide]:
        """
        Get all slides of a specific type.

        Args:
            slide_type: The slide type to filter by

        Returns:
            List of slides matching the specified type
        """
        return [slide for slide in self.slides if slide.slide_type == slide_type]


# Example LLM output schema (for prompt):
# [
#   {
#     "title": "Lesson Objectives",
#     "content": "",
#     "bullet_points": ["**Understand** government structure", "**Analyze** political systems"],
#     "slide_type": "objectives",
#     "notes": "Emphasize the action verbs"
#   },
#   {
#     "title": "Key Concepts",
#     "content": "Democracy is a form of government where power rests with the people.",
#     "bullet_points": ["Popular sovereignty", "Rule of law", "Individual rights"],
#     "slide_type": "content"
#   }
# ]
