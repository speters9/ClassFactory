"""
BeamerBot Module
================

A multi-format presentation slide generator supporting LaTeX Beamer and PowerPoint output.

Main Classes:
-------------
- BeamerBot: Factory class that creates the appropriate slide generator
- LatexSlideGenerator: Generates LaTeX Beamer presentations
- PptxSlideGenerator: Generates PowerPoint presentations [coming soon]
- BaseSlideGenerator: Abstract base class for all slide generators

Supporting Classes:
-------------------
- LatexSlides, LatexSlide: Data models for LaTeX slides
- BeamerSlides, Slide: Backward-compatible aliases for LaTeX slides

Usage:
------
    >>> from class_factory.beamer_bot import BeamerBot
    >>>
    >>> # Create LaTeX generator (default)
    >>> bot = BeamerBot(lesson_no=10, llm=llm, lesson_loader=loader, course_name="Politics")
    >>>
    >>> # Or explicitly specify format
    >>> bot = BeamerBot(output_format="pptx", lesson_no=10, llm=llm, lesson_loader=loader, course_name="Politics")
    >>>
    >>> # Generate and save slides
    >>> slides = bot.generate_slides()
    >>> bot.save_slides(slides)
"""

from class_factory.beamer_bot.base_slide_generator import BaseSlideGenerator
from class_factory.beamer_bot.BeamerBot import BeamerBot
from class_factory.beamer_bot.latex_slide_generator import LatexSlideGenerator
from class_factory.beamer_bot.latex_slides import (BeamerSlides, LatexSlide,
                                                   LatexSlides, Slide)
from class_factory.beamer_bot.pptx_slide_generator import PptxSlideGenerator
from class_factory.beamer_bot.pptx_slides import PptxSlide, PptxSlides

__all__ = [
    "BeamerBot",
    "BaseSlideGenerator",
    "LatexSlideGenerator",
    "PptxSlideGenerator",
    "LatexSlides",
    "LatexSlide",
    "PptxSlides",
    "PptxSlide",
    "BeamerSlides",  # Backward compatibility
    "Slide",  # Backward compatibility
]
