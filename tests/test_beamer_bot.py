"""
Tests for the refactored BeamerBot with Factory pattern and multi-format support.

This test suite covers:
- BeamerBot factory pattern
- LatexSlideGenerator
- PptxSlideGenerator
- Slide model classes (LatexSlides, PptxSlides)
- Base functionality in BaseSlideGenerator
"""

import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from class_factory.beamer_bot.base_slide_generator import BaseSlideGenerator
from class_factory.beamer_bot.BeamerBot import BeamerBot
from class_factory.beamer_bot.latex_slide_generator import LatexSlideGenerator
from class_factory.beamer_bot.latex_slides import LatexSlide, LatexSlides
from class_factory.beamer_bot.pptx_slide_generator import PptxSlideGenerator
from class_factory.beamer_bot.pptx_slides import PptxSlide, PptxSlides
from class_factory.utils.load_documents import LessonLoader

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    return Mock()


@pytest.fixture
def mock_paths():
    """Create temporary directories for testing."""
    temp_dirs = {
        "syllabus_path": Path(tempfile.mkdtemp()) / "syllabus.txt",
        "reading_dir": Path(tempfile.mkdtemp()),
        "slide_dir": Path(tempfile.mkdtemp()),
        "output_dir": Path(tempfile.mkdtemp()),
    }
    temp_dirs["syllabus_path"].write_text("Syllabus content here.")

    yield temp_dirs

    # Cleanup
    for temp_dir in temp_dirs.values():
        if temp_dir.is_dir():
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_lesson_loader(mock_paths):
    """Create a mock LessonLoader with test data."""
    lesson_dir = mock_paths["reading_dir"] / "L1"
    lesson_dir.mkdir(parents=True)
    (lesson_dir / "reading1.txt").write_text("Reading 1 content")
    (lesson_dir / "reading2.txt").write_text("Reading 2 content")

    return LessonLoader(
        syllabus_path=mock_paths["syllabus_path"],
        reading_dir=mock_paths["reading_dir"],
        slide_dir=mock_paths["slide_dir"]
    )


# ============================================================================
# BeamerBot Factory Tests
# ============================================================================

class TestBeamerBotFactory:
    """Test the BeamerBot factory pattern."""

    def test_beamerbot_returns_latex_generator_by_default(self, mock_llm, mock_lesson_loader, mock_paths):
        """Test that BeamerBot returns LatexSlideGenerator by default."""
        bot = BeamerBot(
            lesson_no=1,
            llm=mock_llm,
            course_name="Test Course",
            lesson_loader=mock_lesson_loader,
            output_dir=mock_paths["output_dir"]
        )

        assert isinstance(bot, LatexSlideGenerator)
        assert bot.lesson_no == 1
        assert bot.course_name == "Test Course"

    def test_beamerbot_returns_latex_generator_explicit(self, mock_llm, mock_lesson_loader, mock_paths):
        """Test that BeamerBot returns LatexSlideGenerator when explicitly requested."""
        bot = BeamerBot(
            output_format="latex",
            lesson_no=2,
            llm=mock_llm,
            course_name="Test Course",
            lesson_loader=mock_lesson_loader,
            output_dir=mock_paths["output_dir"]
        )

        assert isinstance(bot, LatexSlideGenerator)
        assert bot.lesson_no == 2

    def test_beamerbot_returns_pptx_generator(self, mock_llm, mock_lesson_loader, mock_paths):
        """Test that BeamerBot returns PptxSlideGenerator when requested."""
        bot = BeamerBot(
            output_format="pptx",
            lesson_no=3,
            llm=mock_llm,
            course_name="Test Course",
            lesson_loader=mock_lesson_loader,
            output_dir=mock_paths["output_dir"]
        )

        assert isinstance(bot, PptxSlideGenerator)
        assert bot.lesson_no == 3

    def test_beamerbot_case_insensitive_format(self, mock_llm, mock_lesson_loader, mock_paths):
        """Test that output_format is case-insensitive."""
        bot1 = BeamerBot(
            output_format="LATEX",
            lesson_no=1,
            llm=mock_llm,
            course_name="Test Course",
            lesson_loader=mock_lesson_loader,
            output_dir=mock_paths["output_dir"]
        )

        bot2 = BeamerBot(
            output_format="  pptx  ",
            lesson_no=1,
            llm=mock_llm,
            course_name="Test Course",
            lesson_loader=mock_lesson_loader,
            output_dir=mock_paths["output_dir"]
        )

        assert isinstance(bot1, LatexSlideGenerator)
        assert isinstance(bot2, PptxSlideGenerator)

    def test_beamerbot_invalid_format_raises_error(self, mock_llm, mock_lesson_loader, mock_paths):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            BeamerBot(
                output_format="invalid",
                lesson_no=1,
                llm=mock_llm,
                course_name="Test Course",
                lesson_loader=mock_lesson_loader,
                output_dir=mock_paths["output_dir"]
            )

    def test_beamerbot_requires_parameters(self, mock_llm, mock_lesson_loader):
        """Test that required parameters raise errors when missing."""
        with pytest.raises(ValueError, match="lesson_no is required"):
            BeamerBot(
                llm=mock_llm,
                course_name="Test",
                lesson_loader=mock_lesson_loader
            )

        with pytest.raises(ValueError, match="llm is required"):
            BeamerBot(
                lesson_no=1,
                course_name="Test",
                lesson_loader=mock_lesson_loader
            )


# ============================================================================
# LatexSlideGenerator Tests
# ============================================================================

class TestLatexSlideGenerator:
    """Test LatexSlideGenerator functionality."""

    @pytest.fixture
    def latex_generator(self, mock_llm, mock_lesson_loader, mock_paths):
        """Create a LatexSlideGenerator instance."""
        with patch.object(LatexSlideGenerator, '_load_prior_lesson', return_value="Mock prior lesson"):
            return LatexSlideGenerator(
                lesson_no=1,
                llm=mock_llm,
                course_name="Test Course",
                lesson_loader=mock_lesson_loader,
                output_dir=mock_paths["output_dir"]
            )

    def test_initialization(self, latex_generator, mock_paths):
        """Test LatexSlideGenerator initialization."""
        assert latex_generator.lesson_no == 1
        assert latex_generator.course_name == "Test Course"
        assert latex_generator.output_dir == mock_paths["output_dir"]
        assert hasattr(latex_generator, 'chain')
        assert hasattr(latex_generator, 'validator')

    def test_format_readings(self, latex_generator):
        """Test reading formatting for prompts."""
        latex_generator._load_readings = Mock(
            return_value={"1": ["Reading1", "Reading2"]}
        )

        formatted = latex_generator._format_readings_for_prompt()

        assert "Lesson 1, Reading 1:" in formatted
        assert "Reading1" in formatted
        assert "Lesson 1, Reading 2:" in formatted
        assert "Reading2" in formatted

    def test_generate_prompt(self, latex_generator):
        """Test prompt generation."""
        prompt = latex_generator._generate_prompt()

        assert prompt is not None
        assert len(prompt.messages) == 2
        assert "{objectives}" in prompt.messages[1].prompt.template
        assert "{information}" in prompt.messages[1].prompt.template
        assert "LaTeX" in prompt.messages[0].content


# ============================================================================
# PptxSlideGenerator Tests
# ============================================================================

class TestPptxSlideGenerator:
    """Test PptxSlideGenerator functionality."""

    @pytest.fixture
    def pptx_generator(self, mock_llm, mock_lesson_loader, mock_paths):
        """Create a PptxSlideGenerator instance."""
        with patch.object(PptxSlideGenerator, '_load_prior_lesson', return_value="Mock prior lesson"):
            return PptxSlideGenerator(
                lesson_no=1,
                llm=mock_llm,
                course_name="Test Course",
                lesson_loader=mock_lesson_loader,
                output_dir=mock_paths["output_dir"]
            )

    def test_initialization(self, pptx_generator, mock_paths):
        """Test PptxSlideGenerator initialization."""
        assert pptx_generator.lesson_no == 1
        assert pptx_generator.course_name == "Test Course"
        assert pptx_generator.output_dir == mock_paths["output_dir"]
        assert hasattr(pptx_generator, 'chain')
        assert hasattr(pptx_generator, 'validator')

    def test_generate_prompt(self, pptx_generator):
        """Test PowerPoint prompt generation."""
        prompt = pptx_generator._generate_prompt()

        assert prompt is not None
        assert len(prompt.messages) == 2
        assert "{objectives}" in prompt.messages[1].prompt.template
        assert "PowerPoint" in prompt.messages[0].content
        # Check that the prompt instructs NOT to use LaTeX
        assert "NOT use LaTeX" in prompt.messages[1].prompt.template or "NO LaTeX" in prompt.messages[1].prompt.template

    def test_strip_markdown(self, pptx_generator):
        """Test markdown stripping utility."""
        text = "This is **bold** and *italic* text"
        result = pptx_generator._strip_markdown(text)

        assert result == "This is bold and italic text"
        assert "**" not in result
        assert "*" not in result


# ============================================================================
# Slide Model Tests
# ============================================================================

class TestLatexSlideModels:
    """Test LatexSlide and LatexSlides models."""

    def test_latex_slide_creation(self):
        """Test creating a LatexSlide."""
        slide = LatexSlide(
            title="Test Slide",
            content="\\begin{itemize}\\item Test\\end{itemize}",
            slide_type="content"
        )

        assert slide.title == "Test Slide"
        assert slide.content == "\\begin{itemize}\\item Test\\end{itemize}"
        assert slide.slide_type == "content"

    def test_latex_slide_to_latex(self):
        """Test converting LatexSlide to LaTeX."""
        slide = LatexSlide(
            title="Test Slide",
            content="Test content",
            slide_type="content"
        )

        latex = slide.to_latex()

        assert "\\begin{frame}" in latex
        assert "Test Slide" in latex
        assert "Test content" in latex
        assert "\\end{frame}" in latex

    def test_latex_slide_titlepage(self):
        """Test titlepage slide special handling."""
        slide = LatexSlide(
            title="Title",
            content="",
            slide_type="titlepage"
        )

        latex = slide.to_latex()

        assert "\\titlepage" in latex
        assert "\\begin{frame}" in latex

    def test_latex_slides_collection(self):
        """Test LatexSlides collection."""
        slides = LatexSlides(
            slides=[
                LatexSlide(title="Slide 1", content="Content 1", slide_type="content"),
                LatexSlide(title="Slide 2", content="Content 2", slide_type="content")
            ],
            title="Lesson 1",
            author="Test Author",
            institute="Test Institute"
        )

        assert len(slides.slides) == 2
        assert slides.title == "Lesson 1"
        assert slides.author == "Test Author"

    def test_latex_slides_to_latex(self):
        """Test converting LatexSlides to full LaTeX document."""
        slides = LatexSlides(
            slides=[
                LatexSlide(title="Slide 1", content="Content 1", slide_type="content")
            ],
            title="Lesson 1",
            author="Test Author",
            institute="Test Institute",
            date="2024-01-01"
        )

        latex = slides.to_latex()

        assert "\\title{Lesson 1}" in latex
        assert "\\author{Test Author}" in latex
        assert "\\institute{Test Institute}" in latex
        assert "\\begin{document}" in latex
        assert "\\end{document}" in latex


class TestPptxSlideModels:
    """Test PptxSlide and PptxSlides models."""

    def test_pptx_slide_creation(self):
        """Test creating a PptxSlide."""
        slide = PptxSlide(
            title="Test Slide",
            content="Main content",
            bullet_points=["Point 1", "Point 2"],
            slide_type="content",
            notes="Speaker notes"
        )

        assert slide.title == "Test Slide"
        assert slide.content == "Main content"
        assert len(slide.bullet_points) == 2
        assert slide.notes == "Speaker notes"

    def test_pptx_slide_to_pptx_data(self):
        """Test converting PptxSlide to data dict."""
        slide = PptxSlide(
            title="Test Slide",
            content="Main content",
            bullet_points=["Point 1"],
            slide_type="content"
        )

        data = slide.to_pptx_data()

        assert data["title"] == "Test Slide"
        assert data["content"] == "Main content"
        assert data["bullet_points"] == ["Point 1"]
        assert data["slide_type"] == "content"

    def test_pptx_slides_collection(self):
        """Test PptxSlides collection."""
        slides = PptxSlides(
            slides=[
                PptxSlide(title="Slide 1", content="Content 1"),
                PptxSlide(title="Slide 2", content="Content 2")
            ],
            title="Lesson 1",
            author="Test Author",
            institute="Test Institute"
        )

        assert len(slides.slides) == 2
        assert slides.title == "Lesson 1"
        assert slides.get_slide_count() == 2

    def test_pptx_slides_to_pptx_data(self):
        """Test converting PptxSlides to full data structure."""
        slides = PptxSlides(
            slides=[
                PptxSlide(title="Slide 1", content="Content 1")
            ],
            title="Lesson 1",
            author="Test Author",
            institute="Test Institute"
        )

        data = slides.to_pptx_data()

        assert "metadata" in data
        assert data["metadata"]["title"] == "Lesson 1"
        assert data["metadata"]["author"] == "Test Author"
        assert "slides" in data
        assert len(data["slides"]) == 1

    def test_pptx_slides_get_by_type(self):
        """Test filtering slides by type."""
        slides = PptxSlides(
            slides=[
                PptxSlide(title="Objectives", slide_type="objectives"),
                PptxSlide(title="Content 1", slide_type="content"),
                PptxSlide(title="Content 2", slide_type="content"),
                PptxSlide(title="Summary", slide_type="summary")
            ],
            title="Lesson 1"
        )

        content_slides = slides.get_slides_by_type("content")

        assert len(content_slides) == 2
        assert all(s.slide_type == "content" for s in content_slides)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch('class_factory.beamer_bot.latex_utils.validate_latex')
    @patch('class_factory.utils.load_documents.LessonLoader.extract_lesson_objectives')
    def test_latex_generation_workflow(
        self,
        mock_extract_obj,
        mock_validate_latex,
        mock_llm,
        mock_lesson_loader,
        mock_paths
    ):
        """Test complete LaTeX generation workflow."""
        mock_extract_obj.return_value = "Test objectives"
        mock_validate_latex.return_value = True

        # Create mock slides
        mock_slides = LatexSlides(
            slides=[LatexSlide(title="Test", content="Content", slide_type="content")],
            title="Lesson 1",
            author="Author"
        )

        # Setup mock chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_slides

        bot = BeamerBot(
            lesson_no=1,
            llm=mock_llm,
            course_name="Test Course",
            lesson_loader=mock_lesson_loader,
            output_dir=mock_paths["output_dir"]
        )

        bot.chain = mock_chain

        with patch.object(bot, '_validate_llm_response', return_value={"status": 1}):
            result = bot.generate_slides()

        assert "\\begin{document}" in result
        assert "\\title{Lesson 1}" in result
        mock_chain.invoke.assert_called_once()

    @patch('class_factory.utils.load_documents.LessonLoader.extract_lesson_objectives')
    def test_pptx_generation_workflow(
        self,
        mock_extract_obj,
        mock_llm,
        mock_lesson_loader,
        mock_paths
    ):
        """Test complete PowerPoint generation workflow."""
        mock_extract_obj.return_value = "Test objectives"

        # Create mock slides
        mock_slides = PptxSlides(
            slides=[PptxSlide(title="Test", content="Content")],
            title="Lesson 1",
            author="Author"
        )

        # Setup mock chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_slides

        bot = BeamerBot(
            output_format="pptx",
            lesson_no=1,
            llm=mock_llm,
            course_name="Test Course",
            lesson_loader=mock_lesson_loader,
            output_dir=mock_paths["output_dir"]
        )

        bot.chain = mock_chain

        with patch.object(bot, '_validate_llm_response', return_value={"status": 1}):
            result = bot.generate_slides()

        assert isinstance(result, PptxSlides)
        assert result.title == "Lesson 1"
        mock_chain.invoke.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
