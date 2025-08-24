"""
Document Loading and Processing Module
-------------------------------------

This module provides functionality to load, process, and extract text from various document types
(PDF, DOCX, and TXT) for generating lesson-specific content. It includes support for OCR processing
of scanned documents and handling of structured educational materials like syllabi and lesson readings.

Classes
~~~~~~~

LessonLoader
    Main class for handling document loading and processing operations.

Key Functions
~~~~~~~

The LessonLoader class provides these key functionalities:

- Document Loading:
    - load_directory: Load all documents from a specified directory
    - load_lessons: Load lessons using the reading index (replaces folder search)
    - load_readings: Extract text from individual documents
    - load_beamer_presentation: Load Beamer presentation content

- Syllabus Processing:
    - extract_lesson_objectives: Extract objectives for specific lessons
    - _load_or_cache_syllabus: Load and parse syllabus content
    - find_docx_indices: Locate lesson sections within syllabus
    - extract_lesson_readings_from_syllabus: Parse lesson reading lists (title/author lines)

- Text Extraction:
    - extract_text_from_pdf: Extract text from PDF files
    - ocr_pdf: Perform OCR on scanned documents

Dependencies
~~~~~~~~~~~

Core Dependencies:
    - pypdf: PDF text extraction
    - python-docx: DOCX file handling
    - pathlib: File path operations
    - typing: Type hints

Optional OCR Dependencies:
    - pytesseract: OCR processing
    - pdf2image: PDF to image conversion

Notes
~~~~~

- OCR functionality requires additional packages via `pip install class_factory[ocr]`
- Folder *naming* is no longer required for lesson detection; we index readings and map by syllabus
"""
import json
import logging
import re
import string
import time
import traceback
import unicodedata
# %%
from importlib.metadata import files
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pypdf
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from markitdown import (FileConversionException, MarkItDown,
                        UnsupportedFormatException)
from pyprojroot.here import here
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from class_factory.utils.ocr_pdf_files import ocr_pdf
from class_factory.utils.prompt_user_for_reading_matches import (
    prompt_user_assign_unmatched, prompt_user_for_reading_matches)
from class_factory.utils.tools import logger_setup, normalize_unicode

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})

try:
    import pytesseract
    from docling.document_converter import DocumentConverter
    from pdf2image import convert_from_path
    from PIL import Image
    from textblob import TextBlob
except ImportError:
    pytesseract = None
    Image = None
    convert_from_path = None
    TextBlob = None
    DocumentConverter = None

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


STOPWORDS = {
    # keep tiny, just the usual clutter; we’re not doing NLP here
    "the", "a", "an", "of", "in", "on", "for", "to", "with", "without",
    "and", "or", "by", "at", "from", "into", "about", "vs", "versus"
}


# ------------------------- Main Classes: Reading Indexer -------------------------
# --- ReadingIndexer: Handles all reading-to-lesson indexing logic ---
class ReadingIndexer:
    def __init__(self, reading_dir: Union[Path, str], syllabus: List[str],
                 tabular_syllabus: bool = False, index_similarity: float = 0.8,
                 initial_index: Optional[dict] = None):
        self.reading_dir = Path(reading_dir)
        self.syllabus = syllabus
        self.tabular_syllabus = tabular_syllabus
        self.index_similarity = index_similarity
        self.logger = logger_setup(logger_name="reading_indexer", log_level=logging.INFO)
        self.reading_index = initial_index if initial_index is not None else {}

    @staticmethod
    def _invoke_update(callback, idx, phase: str):
        try:
            callback(idx, phase)
        except TypeError:
            # Back-compat: old callbacks that only take (idx)
            callback(idx)

    def build_or_update_reading_index(self, current_index: Optional[dict] = None, prompt_user: bool = True, on_update_index=None) -> dict:
        """
        Build or update the lesson-centric index in memory. Optionally prompt user for feedback on low-confidence matches.
        Returns the updated index dict.
        If on_update_index is provided, it will be called with the updated index after user feedback or immediately if the index already exists.
        """
        if self.reading_index:
            self.logger.info("Using existing index…")
            if on_update_index:
                self._invoke_update(on_update_index, self.reading_index, phase="loaded_existing")
            return self.reading_index

        self.logger.info("No pre-existing index found. Generating a new one…")
        index = self.build_lesson_centric_index(existing_index=current_index)

        if prompt_user:
            try:
                self._user_feedback_received = False

                def on_submit_callback(user_choices, lesson_index):
                    for lesson, choices in user_choices.items():
                        if lesson in index:
                            for reading, assigned_file in choices.items():
                                if reading in index[lesson]["readings"]:
                                    index[lesson]["readings"][reading]["assigned_file"] = assigned_file
                    self._user_feedback_received = True
                    # >>> Finalize after user resolves low-confidence matches
                    if on_update_index:
                        self._invoke_update(on_update_index, index, phase="finalized")

                prompt_user_for_reading_matches(
                    index,
                    match_threshold=self.index_similarity,
                    top_n=3,
                    on_submit_callback=on_submit_callback
                )
                return index  # callback will fire upon user submit
            except ImportError:
                self.logger.warning("ipywidgets not available; skipping interactive reading assignment.")
            except Exception as e:
                self.logger.warning(f"Error during interactive reading assignment: {e}")

        # No GUI path (or it failed): finalize immediately so downstream can proceed
        if on_update_index:
            self._invoke_update(on_update_index, index, phase="finalized")
        return index

    @staticmethod
    def _ascii_fold(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    @staticmethod
    def _squash_spaces(s: str) -> str:
        return re.sub(r"\s{2,}", " ", s).strip()

    def normalize_for_match(self, s: str) -> str:
        """
        Canonicalize arbitrary text for fuzzy matching.
        - unicode → ascii
        - strip markdown/formatting
        - drop years, page/chap refs, parenthetical guidance
        - remove punctuation
        - collapse/trim whitespace
        - remove short/common stopwords
        """
        if not s:
            return ""

        # course-specific unicode normalizer if you have one
        try:
            s = normalize_unicode(s)
        except Exception:
            pass

        s = self._ascii_fold(s)

        # strip markdown emphasis (**bold**, *italics*)
        s = re.sub(r"[*_`]+", " ", s)

        # remove parenthetical guidance like (Read ...), (Skim ...), (pp ...), bare (…) years
        s = re.sub(r"\((?:Read|Skim)[^)]+\)", " ", s, flags=re.I)
        s = re.sub(r"\bpp?\.?\s*\d+(?:\s*[-–]\s*\d+)?", " ", s, flags=re.I)
        s = re.sub(r"\bchap(?:ter)?\.?\s*\d+(?:\s*[-–]\s*\d+)?", " ", s, flags=re.I)
        s = re.sub(r"\bch\.?\s*\d+(?:\s*[-–]\s*\d+)?", " ", s, flags=re.I)
        s = re.sub(r"\b\d{4}\b", " ", s)  # drop standalone years

        # normalise joiners
        s = re.sub(r"\s+(?:&|\+)\s+", " and ", s)

        # kill remaining punctuation and pipes
        s = s.replace("|", " ").translate(_PUNCT_TABLE)

        # collapse whitespace
        s = self._squash_spaces(s.lower())

        # tokenize → drop stopwords and 1-char tokens
        tokens = [t for t in s.split() if len(t) > 1 and t not in STOPWORDS]

        # sort & dedupe tokens to make order irrelevant (helps against “token_set_ratio” quirks)
        # you can skip sorting if you prefer original order; this makes the key deterministic.
        tokens = sorted(set(tokens))

        return " ".join(tokens)

    def extract_lesson_readings_from_linear(self, syllabus_lines: list) -> Dict[int, List[dict]]:
        """
        STRICT mode for linear syllabi.

        Improvements vs previous version:
        - Accept Markdown headings before 'Lesson'/'Week' (e.g., '## Lesson 3 : …').
        - Robust 'Reading' labels in bullets ('- Reading :', '- Reading:', etc.), with synonyms optional.
        - Treat 'TBD', 'No reading', 'None' as empty (no reading) for that lesson block.
        - Split list items on bullets *and* enumerations (1., a), i.), not just semicolons.
        - Accept 'pg', 'pgs', 'pages', 'p.', 'pp.' for page spans.
        """
        lines: List[str] = syllabus_lines
        readings_by_lesson: Dict[int, List[dict]] = {}

        # ---------------- regexes ----------------
        # Markdown heading optional, then Lesson/Week + number
        RE_LESSON_HDR = re.compile(
            r'^\s*(?:#{1,6}\s*)?(?:Lesson|Week)\s*([0-9]{1,3})\b', re.I
        )

        # Common section breaks (kept conservative — add more if you see collisions)
        BREAK_ALIASES = [
            r"Objectives", r"Learning\s+Objectives", r"Topics", r"Agenda", r"Activities",
            r"Due", r"Quiz", r"Exam", r"Notes", r"Discussion", r"Policies?", r"Homework",
            r"Deliverables?", r"Project", r"Lab", r"Grading", r"Schedule", r"Resources?",
            r"Assignments?"
        ]
        RE_SECTION_BREAK = re.compile(r'^\s*(?:' + '|'.join(BREAK_ALIASES) + r')\b', re.I)

        # Bullets and enumerations treated as item starts
        RE_ITEM_START = re.compile(
            r'(?m)^\s*(?:[*\-•·▪]|\(?\d+\)?[.)]|[A-Za-z][.)]|\(?[ivxlcdm]+\)?[.)])\s+'
        )
        RE_SEMIS = re.compile(r'\s*;\s*')

        # Reading labels you’ll accept on bullets (first word only — avoids swallowing 'Assignment')
        READING_ALIASES = [r"Reading", r"Required\s*Reading", r"Text", r"Texts?", r"Materials?"]
        RE_BULLET_READING_LABEL = re.compile(
            # optional bullet or enumeration at line start, then a Reading label
            r'(?m)^\s*(?:[*\-•·▪]|\(?\d+\)?[.)]|[A-Za-z][.)]|\(?[ivxlcdm]+\)?[.)])?\s*'
            r'(?:' + '|'.join(READING_ALIASES) + r')\s*:?\s*(.*)$',
            re.I
        )

        # "None"/"TBD"/"No reading"
        RE_NONE = re.compile(r'\b(?:no\s+readings?|none|t\.?b\.?d\.?|tba|n/?a)\b', re.I)

        # -------------- helpers -----------------
        def _clean(s: str) -> str:
            s = s.replace('|', ' ')
            s = re.sub(r'\s{2,}', ' ', s)
            s = re.sub(r'(\w)-\s+(\w)', r'\1\2', s)  # unwrap soft hyphens
            s = re.sub(r'^\s*([*\-•·▪]|\(?[0-9ivxlcdm]+\)?[.)]|[A-Za-z][.)])\s+', '', s)
            return s.strip()

        def _strict_split(text: str) -> List[str]:
            # Treat bullets/enumerations as hard line breaks, then split on semicolons.
            text = RE_ITEM_START.sub('\n', text)
            # inline " * " → newline
            text = re.sub(r'\s\*\s+(?=\S)', '\n', text)
            out: List[str] = []
            for raw in text.split('\n'):
                raw = raw.strip()
                if not raw:
                    continue
                out.extend(p.strip() for p in RE_SEMIS.split(raw) if p.strip())
            # drop tiny fragments
            return [_clean(x) for x in out if len(re.sub(r'\s+', '', x)) >= 2]

        def _parse_pages_and_guidance(item: str):
            # accept p., pp., pg, pgs, pages (with/without dot)
            pages = None
            m = re.search(r'\b(?:pp?|pgs?|pages?)\.?\s*([0-9]+(?:\s*[-–]\s*[0-9]+)?)', item, re.I)
            if m:
                pages = m.group(1)
                item = re.sub(r'\b(?:pp?|pgs?|pages?)\.?\s*[0-9]+(?:\s*[-–]\s*[0-9]+)?', '', item, flags=re.I)
            # remove (Read ...)/(Skim ...) parentheticals
            item = re.sub(r'\((?:Read|Skim)[^)]+\)', '', item, flags=re.I).strip(' ,;.-')
            return item, pages

        def _parse_reading(item: str) -> dict:
            original = item
            item, pages = _parse_pages_and_guidance(item)

            # Author (Year): Title
            m = re.match(r'^(?P<authors>.+?)\s*\((?P<year>\d{4})\)\s*[:–-]\s*(?P<title>.+)$', item)
            if m:
                authors = [a.strip() for a in re.split(r'\band\b|&|,', m.group('authors')) if a.strip()]
                title = m.group('title').strip().strip('"“”')
                return {"raw": original, "authors": authors or None, "year": int(m.group('year')),
                        "title": title or None, "work": None, "chapter": None, "pages": pages, "assigned": True}

            # Author, Work, Chapter n
            m = re.match(r'^(?P<authors>[^,]+),\s*(?P<work>.+?)\s*(?:,|\s)\bChap(?:\.|ter)?s?\s*(?P<chap>\d+)\b.*$', item, re.I)
            if m:
                authors = [a.strip() for a in re.split(r'\band\b|&|,', m.group('authors')) if a.strip()]
                return {"raw": original, "authors": authors or None, "year": None, "title": None,
                        "work": m.group('work').strip(), "chapter": int(m.group('chap')),
                        "pages": pages, "assigned": True}

            # Author, Work (no chapter)
            m = re.match(r'^(?P<authors>[^,]+)\s*,\s*(?P<work>.+)$', item)
            if m:
                authors = [a.strip() for a in re.split(r'\band\b|&|,', m.group('authors')) if a.strip()]
                return {"raw": original, "authors": authors or None, "year": None, "title": None,
                        "work": m.group('work').strip(), "chapter": None, "pages": pages, "assigned": True}

            # Fallback to a title-like string
            return {"raw": original, "authors": None, "year": None,
                    "title": item.strip('"“”'), "work": None, "chapter": None, "pages": pages, "assigned": True}

        # ---- locate lesson bounds
        starts: List[Tuple[int, int]] = []
        for i, raw in enumerate(lines):
            s = _clean(raw)
            m = RE_LESSON_HDR.match(s)
            if m:
                starts.append((i, int(m.group(1))))
        starts.sort(key=lambda t: t[0])

        bounds: List[Tuple[int, int, int]] = []
        for idx, (si, num) in enumerate(starts):
            ei = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
            bounds.append((num, si, ei))

        # ---- per lesson parse
        for lesson_num, si, ei in bounds:
            block_lines = [lines[k] for k in range(si, ei)]
            cleaned_block = [_clean(b) for b in block_lines]

            items: List[str] = []
            in_reading = False
            buffer: List[str] = []

            def _flush():
                if buffer:
                    # If buffer declares no reading/TBD, drop it
                    if not RE_NONE.search(" ".join(buffer)):
                        items.extend(_strict_split("\n".join(buffer)))
                    buffer.clear()

            # Primary path: explicit "Readings" header (rare in your sample but keep it)
            RE_READING_HDR = re.compile(r'^\s*Readings?\s*:?\s*(.*)$', re.I)
            for raw in cleaned_block:
                if not raw:
                    if in_reading:
                        buffer.append('')
                    continue

                m_read = RE_READING_HDR.match(raw)
                if m_read:
                    if in_reading:
                        _flush()
                    in_reading = True
                    tail = m_read.group(1).strip()
                    if tail:
                        buffer.append(tail)
                    continue

                # any section break ends a reading section
                if in_reading and (RE_SECTION_BREAK.match(raw) or RE_READING_HDR.match(raw)):
                    _flush()
                    in_reading = False
                    continue

                if in_reading:
                    buffer.append(raw)

            if in_reading:
                _flush()

            # Fallback path: bullet lines labeled "Reading:" (covers your sample)
            if not items:
                # Use raw lines so leading "-" is still there for anchoring & cutting
                joined = "\n".join(block_lines)
                matches = list(RE_BULLET_READING_LABEL.finditer(joined))
                for mi, m in enumerate(matches):
                    start = m.end()  # right after the label capture
                    seed = (m.group(1) or "").strip()

                    # stop at the start of the next reading label OR end of this lesson block
                    stop = matches[mi + 1].start() if mi + 1 < len(matches) else len(joined)
                    segment = joined[start:stop]

                    # Do not swallow later bullets/enumerations/headers in this chunk
                    # (cut at the first next item start)
                    cut = re.split(RE_ITEM_START, segment, maxsplit=1)
                    segment = cut[0]

                    candidate = "\n".join([p for p in [seed, segment.strip()] if p])
                    if candidate and not RE_NONE.search(candidate):
                        items.extend(_strict_split(candidate))

            # Build structured output
            readings_by_lesson[lesson_num] = [_parse_reading(it) for it in items]

        return readings_by_lesson

    def extract_lesson_readings_from_tabular(self, syllabus_lines: list) -> Dict[int, List[dict]]:
        """
        STRICT mode for table syllabi:
        | Lesson | ... | Reading/Assignment |
        Inside each Reading/Assignment cell, readings must be separated by
        - bullets at the start of a line (*, -, •, etc.), or
        - semicolons (;).
        """

        lines: List[str] = syllabus_lines
        readings_by_lesson: Dict[int, List[dict]] = {}

        BULLET_START = re.compile(r'(?m)^\s*([*\-•·▪])\s+')
        SEMIS = re.compile(r'\s*;\s*')

        def _clean(s: str) -> str:
            s = s.replace('|', ' ')                # pipes are table artifacts
            s = re.sub(r'\s{2,}', ' ', s)
            s = re.sub(r'(\w)-\s+(\w)', r'\1\2', s)  # undo soft-wrap hyphens
            # remove any leading bullet/enumerator that slipped through
            s = re.sub(r'^\s*([*\-•·▪]|\(?[0-9ivxlcdm]+\)?[.)]|[A-Za-z][.)])\s+', '', s)
            return s.strip()

        def _parse_pages_and_guidance(item: str):
            pages = None
            m = re.search(r'\bpp?\.?\s*([0-9]+(?:\s*[-–]\s*[0-9]+)?)', item, flags=re.I)
            if m:
                pages = m.group(1)
                item = re.sub(r'\bpp?\.?\s*[0-9]+(?:\s*[-–]\s*[0-9]+)?', '', item, flags=re.I)
            item = re.sub(r'\((?:Read|Skim)[^)]+\)', '', item, flags=re.I).strip(' ,;.-')
            return item, pages

        def _parse_reading(item: str) -> dict:
            original = item
            item, pages = _parse_pages_and_guidance(item)

            m = re.match(r'^(?P<authors>.+?)\s*\((?P<year>\d{4})\)\s*[:–-]\s*(?P<title>.+)$', item)
            if m:
                authors = [a.strip() for a in re.split(r'\band\b|&|,', m.group('authors')) if a.strip()]
                title = m.group('title').strip().strip('"“”')
                return {"raw": original, "authors": authors or None, "year": int(m.group('year')),
                        "title": title or None, "work": None, "chapter": None, "pages": pages, "assigned": True}

            m = re.match(r'^(?P<authors>[^,]+),\s*(?P<work>.+?)\s*(?:,|\s)\bChap(?:\.|ter)?\s*(?P<chap>\d+)\b.*$', item, flags=re.I)
            if m:
                authors = [a.strip() for a in re.split(r'\band\b|&|,', m.group('authors')) if a.strip()]
                return {"raw": original, "authors": authors or None, "year": None, "title": None,
                        "work": m.group('work').strip(), "chapter": int(m.group('chap')),
                        "pages": pages, "assigned": True}

            m = re.match(r'^(?P<authors>[^,]+)\s*,\s*(?P<work>.+)$', item)
            if m:
                authors = [a.strip() for a in re.split(r'\band\b|&|,', m.group('authors')) if a.strip()]
                return {"raw": original, "authors": authors or None, "year": None, "title": None,
                        "work": m.group('work').strip(), "chapter": None, "pages": pages, "assigned": True}

            return {"raw": original, "authors": None, "year": None, "title": item.strip('"“”'),
                    "work": None, "chapter": None, "pages": pages, "assigned": True}

        def _strict_split_cell(cell: str) -> List[str]:
            # Convert inline bullets to newlines, then split by semicolons.
            s = cell.replace('\r', '')
            # turn " * " and similar into newlines
            s = BULLET_START.sub('\n', s)
            # also handle inline bullets like " * **Author**" that aren't at start because of MD quirks
            s = re.sub(r'\s\*\s+(?=\S)', '\n', s)
            # now split on semicolons too
            parts: List[str] = []
            for line in s.split('\n'):
                line = line.strip()
                if not line:
                    continue
                parts.extend(p for p in SEMIS.split(line) if p.strip())
            # clean and filter
            items = [_clean(p) for p in parts if len(re.sub(r'\s+', '', p)) >= 2]
            return items

        # ---- find header and columns ----
        header_idx: Optional[int] = None
        header_cols: Optional[List[str]] = None
        lesson_col = reading_col = None

        for i, line in enumerate(lines):
            if line.strip().startswith("|") and re.search(r'\b(lesson|week)\b', line, re.I) and re.search(r'\b(reading|assignment|text)\b', line, re.I):
                header_idx = i
                header_cols = [h.strip().lower() for h in line.split("|") if h.strip()]
                break
        if header_idx is None or not header_cols:
            return {}

        for idx, col in enumerate(header_cols):
            if lesson_col is None and re.search(r'\b(lesson|week)\b', col):
                lesson_col = idx
            if reading_col is None and re.search(r'\b(reading|assignment|text)\b', col):
                reading_col = idx
        if lesson_col is None or reading_col is None:
            return {}

        # ---- process rows ----
        for line in lines[header_idx + 1:]:
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().split("|") if c.strip()]
            if len(cells) <= max(lesson_col, reading_col):
                continue

            m = re.search(r'\d{1,3}', cells[lesson_col])
            if not m:
                continue
            lesson_num = int(m.group())

            cell = cells[reading_col]
            items = _strict_split_cell(cell)

            readings_by_lesson.setdefault(lesson_num, []).extend(_parse_reading(it) for it in items)

        return readings_by_lesson

    def extract_lesson_readings_from_syllabus(self) -> Dict[int, List[dict]]:
        """Dispatcher (strict mode). Pass in loaded syllabus lines and tabular_syllabus flag."""
        syllabus_lines = self.syllabus
        use_tabular = self.tabular_syllabus

        if not use_tabular:
            use_tabular = any(
                ln.strip().startswith("|") and
                re.search(r'\b(lesson|week)\b', ln, re.I) and
                re.search(r'\b(reading|assignment|text)\b', ln, re.I)
                for ln in syllabus_lines
            )
        return (self.extract_lesson_readings_from_tabular(syllabus_lines)
                if use_tabular else
                self.extract_lesson_readings_from_linear(syllabus_lines))

    @staticmethod
    def reading_to_string(r):
        bits = []
        if r.get("authors"):
            bits.append(" ".join(r["authors"]))
        if r.get("year"):
            bits.append(str(r["year"]))
        if r.get("work"):
            bits.append(r["work"])
        if r.get("title"):
            bits.append(r["title"])
        if r.get("chapter"):
            bits.append(f"chapter {r['chapter']}")
        if r.get("pages"):
            bits.append(f"pp {r['pages']}")
        return " | ".join(bits) or r.get("raw", "")

    def get_lesson_readings_map(self):
        simple_index = {}

        # NEW: support both {raw_match_results: {...}} and {<lesson>: {...}} shapes
        lessons_dict = self.reading_index.get("raw_match_results", self.reading_index)

        for lesson_key, lesson_entry in lessons_dict.items():
            if not isinstance(lesson_entry, dict):
                continue
            lesson = lesson_entry.get("number", lesson_key)
            files = []
            for _reading, entry in lesson_entry.get("readings", {}).items():
                f = entry.get("assigned_file")
                if f and f != "__NOT_A_READING__":
                    files.append(f)

            # stable dedupe
            seen = set()
            dedup = [f for f in files if not (f in seen or seen.add(f))]
            simple_index[str(lesson)] = dedup

        return {k: v for k, v in simple_index.items() if k != 'raw_match_results'}

    def export_lesson_readings_index(self, syllabus_lines: Optional[list] = None) -> dict:
        """
        Return a merged index dict with:
        - lesson_readings_index (simplified map)
        - syllabus (as lines)
        - raw_match_results (full reading index, all lessons/readings)
        """
        simple_index = self.get_lesson_readings_map()

        lessons_dict = self.reading_index.get("raw_match_results", self.reading_index)
        raw_match_results = dict(lessons_dict)

        merged = {
            "lesson_readings_index": simple_index,
            "raw_match_results": raw_match_results
        }
        if syllabus_lines is not None:
            merged["syllabus"] = syllabus_lines
        return merged

    def build_lesson_centric_index(
        self,
        existing_index: Optional[dict] = None,
        top_n: int = 3,
        similarity_method: str = 'default',  # or 'tfidf' or 'rf', 'ensemble'
        assign_threshold: float = 0.8,
        blend_weights: Tuple[float, float] = (0.6, 0.4),
        min_token_overlap: int = 1
    ) -> dict:
        if fuzz is None:
            raise ImportError("RapidFuzz is required. Install with `pip install rapidfuzz`.")

        def tfidf_clean(s: str) -> str:
            s = normalize_unicode(s)
            s = self._ascii_fold(s)
            s = re.sub(r"[*_`]+", " ", s)
            s = s.replace("|", " ")
            s = re.sub(r"[^\w\s-]+", " ", s)
            s = re.sub(r"\s{2,}", " ", s).strip().lower()
            return s

        files = [
            p for p in self.reading_dir.glob("**/*")
            if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt"}
        ]
        file_entries = []
        for f in files:
            stem = normalize_unicode(f.stem)
            canon = self.normalize_for_match(stem)
            tfidf_text = tfidf_clean(stem)
            file_entries.append({
                "path": f,
                "name": f.name,
                "stem": stem,
                "canon": canon,
                "canon_tokens": set(canon.split()) if canon else set(),
                "tfidf_text": tfidf_text,
            })

        lesson_readings = self.extract_lesson_readings_from_syllabus()
        index = existing_index.copy() if existing_index else {}
        for lesson, reading_dicts in lesson_readings.items():
            lesson_entry = {"number": lesson, "readings": {}}
            corpus_files_tfidf = [fe["tfidf_text"] for fe in file_entries]
            vectorizer = TfidfVectorizer()
            if corpus_files_tfidf:
                vectorizer.fit(corpus_files_tfidf)
            for reading in reading_dicts:
                raw_str = normalize_unicode(self.reading_to_string(reading))
                canon_reading = self.normalize_for_match(raw_str)
                canon_tokens = set(canon_reading.split()) if canon_reading else set()
                tfidf_reading = tfidf_clean(raw_str)

                if len(canon_tokens) < 2:
                    lesson_entry["readings"][raw_str] = {
                        "canon_reading": canon_reading,
                        "tfidf_reading": tfidf_reading,
                        "matches": [],
                        "assigned_file": None,
                        "note": "Skipped: too generic to auto-match"
                    }
                    continue

                tfidf_scores = []
                if corpus_files_tfidf and len(vectorizer.vocabulary_) > 0:
                    r_vec = vectorizer.transform([tfidf_reading])
                    f_mat = vectorizer.transform(corpus_files_tfidf)
                    cos = cosine_similarity(r_vec, f_mat).ravel()
                    tfidf_scores = cos.tolist()
                else:
                    tfidf_scores = [0.0] * len(file_entries)
                matches = []
                for i, fe in enumerate(file_entries):
                    if min_token_overlap > 0 and canon_tokens and fe["canon_tokens"]:
                        if len(canon_tokens.intersection(fe["canon_tokens"])) < min_token_overlap:
                            continue

                    rf_score = fuzz.token_set_ratio(fe["canon"], canon_reading) / 100.0 if (canon_reading or fe["canon"]) else 0.0
                    tf_score = tfidf_scores[i]

                    if similarity_method == 'rf':
                        combined = rf_score
                    elif similarity_method == 'tfidf':
                        combined = tf_score
                    elif similarity_method == 'ensemble':
                        w_rf, w_tf = blend_weights
                        combined = (w_rf * rf_score) + (w_tf * tf_score)
                    else:  # ie 'default'
                        combined = sum([rf_score, tf_score])/2

                    matches.append({
                        "file": fe["name"],
                        "matched_text": fe["stem"],
                        "canon_file": fe["canon"],
                        "rf_score": round(rf_score, 2),
                        "tfidf_score": round(tf_score, 2),
                        "combined_score": round(combined, 2),
                        "overlap_tokens": sorted(canon_tokens.intersection(fe["canon_tokens"])) if canon_tokens else [],
                    })
                matches = sorted(
                    matches,
                    key=lambda x: (x["combined_score"], x["tfidf_score"], x["rf_score"]),
                    reverse=True
                )
                top_matches = matches[:top_n]
                assigned_file = None
                if top_matches and top_matches[0]["combined_score"] >= assign_threshold:
                    assigned_file = top_matches[0]["file"]
                lesson_entry["readings"][raw_str] = {
                    "canon_reading": canon_reading,
                    "tfidf_reading": tfidf_reading,
                    "matches": top_matches,
                    "assigned_file": assigned_file,
                }
            index[lesson] = lesson_entry
        self.reading_index = index
        return index

    def assign_unmatched_readings(self, on_update_callback=None):
        """
        Assign unmatched readings in memory. Calls on_update_callback(updated_index) if provided.
        Uses get_lesson_readings_map to determine assigned files.
        """
        def handle_assignments(assignments, updated_index=None):
            idx = updated_index if updated_index is not None else self.reading_index

            # Work on the canonical container for lessons
            lessons_dict = idx.get("raw_match_results", idx)

            # Normalize lesson keys to strings to match how you later read/write
            # (get_lesson_readings_map uses string keys)
            str_keyed_assignments = {str(k): set(v) for k, v in assignments.items()}

            # Update assigned_file for each reading under the correct container
            for lesson_key, lesson_entry in lessons_dict.items():
                if not isinstance(lesson_entry, dict):
                    continue
                # lesson_key may be int in older data; normalize to string for lookup
                assigned_files = str_keyed_assignments.get(str(lesson_key), set())
                if not assigned_files:
                    continue

                for reading, entry in lesson_entry.get("readings", {}).items():
                    # prefer an explicit match in the ranked matches
                    assigned_file = None
                    for m in entry.get("matches", []):
                        if m.get("file") in assigned_files:
                            assigned_file = m["file"]
                            break
                    # fallbacks
                    if assigned_file is None:
                        if entry.get("assigned_file") in assigned_files:
                            assigned_file = entry["assigned_file"]
                        elif reading in assigned_files:
                            assigned_file = reading
                    entry["assigned_file"] = assigned_file

            # Rebuild the simplified map from the canonical container
            # (get_lesson_readings_map already reads from self.reading_index['raw_match_results'])
            idx["lesson_readings_index"] = self.get_lesson_readings_map()

            # Persist back to self.reading_index
            self.reading_index = idx

            if on_update_callback:
                on_update_callback(self.reading_index)
            print("\nAssignments updated and merged.")

        try:
            lesson_readings = self.extract_lesson_readings_from_syllabus()
            expected_counts = {lesson: len(readings) for lesson, readings in lesson_readings.items()}
            lessons = sorted(expected_counts.keys())

            lesson_readings_map = self.get_lesson_readings_map()
            current_assignments = {str(lesson): files for lesson, files in lesson_readings_map.items()}

            assigned_files = set(
                f for files in lesson_readings_map.values() for f in files if not f.startswith("__NOT_A_READING__")
            )

            # List all files in the reading directory
            all_files = set(
                p.name for p in self.reading_dir.glob("**/*")
                if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt"}
            )

            # Unassigned = files in dir not assigned in index
            unassigned = sorted(all_files - assigned_files)

            if not unassigned:
                print("All readings are assigned.")
                if on_update_callback:
                    on_update_callback(self.reading_index)
                return
            print("The following readings are unassigned and need to be assigned to a lesson. You may assign a reading to multiple lessons if it is used for review or repeat discussion.")
            prompt_user_assign_unmatched(
                unassigned,
                lessons,
                expected_counts,
                current_assignments=current_assignments,
                on_submit_callback=handle_assignments
            )
        except ImportError:
            if self.logger:
                self.logger.warning("ipywidgets not available; skipping unmatched reading assignment.")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error during unmatched reading assignment: {e}")


class LessonLoader:
    """ A class for loading and managing educational content from various document formats.
        This class handles loading and processing of lesson materials including syllabi, readings,
        and Beamer presentations. It supports multiple file formats (PDF, DOCX, TXT)
        and provides OCR capabilities for scanned documents when necessary.
        New in this version: - Course-specific reading indices via class_name - Index-based lesson loading
        (replaces folder-name scanning) - Preserves tabular/non-tabular syllabus parsing and objective
        extraction - verified flag in the index for human-confirmed matches """

    def __init__(self, syllabus_path: Union[Path, str], reading_dir: Union[Path, str], slide_dir: Optional[Union[Path, str]] = None, project_dir: Optional[Union[Path, str]] = None, class_name: Optional[str] = None, verbose: bool = True, tabular_syllabus: bool = False, tabular_lesson_col: Optional[str] = None, tabular_readings_col: Optional[str] = None, index_similarity: float = 0.8):
        """Initialize LessonLoader and build/load a per-course reading index."""
        log_level = logging.INFO if verbose else logging.WARNING
        self.logger = logger_setup(logger_name="lesson_loader_logger", log_level=log_level)
        self.reading_dir = self._validate_dir_path(reading_dir, "reading directory")
        self.slide_dir = self._validate_dir_path(slide_dir, "slide directory") if slide_dir else None
        self.project_dir = self._validate_dir_path(project_dir, "root project directory") if project_dir else here()
        self.syllabus_path = self._validate_file_path(syllabus_path, "syllabus file") if syllabus_path else None
        self.class_name = class_name or "default"
        self.tabular_syllabus = tabular_syllabus
        self.tabular_lesson_col_name = tabular_lesson_col
        self.tabular_readings_col_name = tabular_readings_col
        self.tabular_lesson_col_idx = None
        self.tabular_readings_col_idx = None
        self.index_similarity = index_similarity
        if not syllabus_path:
            self.logger.warning("No syllabus path provided. Syllabus-driven reading matching will be limited.")
        # Per-course index file
        self.index_path = self.project_dir / f"reading_index_{self.class_name}.json"
        self.index_exists = self.index_path.exists()

        self._index_json = self._read_index_json() if self.index_exists else {}
        self.syllabus_lines = self._load_or_cache_syllabus(self.syllabus_path)
        self.indexer = ReadingIndexer(
            reading_dir=self.reading_dir,
            syllabus=self.syllabus_lines,
            tabular_syllabus=self.tabular_syllabus,
            index_similarity=self.index_similarity,
            initial_index=self._index_json
        )
        self.reading_index: Dict[str, Dict] = self.indexer.build_or_update_reading_index(
            on_update_index=self._save_index_check_unmatched_callback
        )

    def _save_index_check_unmatched_callback(self, updated_index, phase: str = "finalized"):
        self.reading_index = updated_index
        self.indexer.reading_index = updated_index
        self.write_simplified_index()

        # Run the second GUI only when it makes sense
        if phase in ("finalized", "loaded_existing"):
            self.indexer.assign_unmatched_readings(on_update_callback=self._save_index_callback)

    def _save_index_callback(self, updated_index):
        self.reading_index = updated_index
        self.write_simplified_index()

    def _read_index_json(self) -> dict:
        try:
            if self.index_path.exists():
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # If old format, upgrade to new format
                    if "raw_match_results" not in data and any(isinstance(v, dict) and "readings" in v for v in data.values()):
                        # Move all lesson dicts to raw_match_results
                        lesson_keys = [k for k, v in data.items() if isinstance(v, dict) and "readings" in v]
                        raw_match_results = {k: data[k] for k in lesson_keys}
                        for k in lesson_keys:
                            data.pop(k)
                        data["raw_match_results"] = raw_match_results
                    return data
        except Exception:
            pass
        return {}

    def _write_index_json(self, new_data: dict) -> None:
        """
        Private: Only called by write_simplified_index. Merges lesson_readings_index and raw_match_results by union, preserving all other data.
        """
        try:
            # Read the existing file if it exists
            if self.index_path.exists():
                with open(self.index_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = {}
            # Merge lesson_readings_index
            old_lri = existing.get("lesson_readings_index", {})
            new_lri = new_data.get("lesson_readings_index", {})
            for lesson, new_readings in new_lri.items():
                old_readings = set(old_lri.get(lesson, []))
                merged = list(old_readings.union(new_readings))
                old_lri[lesson] = merged
            existing["lesson_readings_index"] = old_lri
            # Merge raw_match_results (overwrite per-lesson, but preserve other keys)
            if "raw_match_results" in new_data:
                if "raw_match_results" not in existing:
                    existing["raw_match_results"] = {}
                for lesson, lesson_data in new_data["raw_match_results"].items():
                    existing["raw_match_results"][lesson] = lesson_data
            # Optionally update syllabus or other keys as needed
            if "syllabus" in new_data:
                existing["syllabus"] = new_data["syllabus"]
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not write index to {self.index_path}: {e}")

    def write_simplified_index(self):
        """
        Write the merged index (lesson_readings_index, syllabus, raw_match_results) to the index JSON file.
        This is the ONLY public way to write the index file.
        """
        merged = self.indexer.export_lesson_readings_index(syllabus_lines=self.syllabus_lines)
        self._write_index_json(merged)

    def get_assigned_files_for_lesson(self, lesson_no: int) -> List[str]:
        lri = self.reading_index.get("lesson_readings_index")
        if lri is None:
            # compute on demand from current index, then cache it in-memory
            lri = self.indexer.get_lesson_readings_map()
            self.reading_index["lesson_readings_index"] = lri
            # persist immediately if it was missing
            self.write_simplified_index()

        # stable dedupe just in case
        files = list(dict.fromkeys(lri.get(str(lesson_no), [])))  # stable dedupe

        if files:
            pretty = "\n  - " + "\n  - ".join(files)
            self.logger.info(
                f"""Lesson {lesson_no}: using the following readings:\n{
                    pretty}\n\nIf this is incorrect, visit the course index to update the filenames for this lesson's readings. The index is located here:\n{str(self.index_path)}"""
            )
        else:
            self.logger.warning(
                f"""Lesson {lesson_no}: no readings assigned. If this is unexpected, adjust the correct filenames within the course index.
                Index is located here:\n{str(self.index_path)}."""
            )
        return files

    def get_reading_texts_by_lesson(self, lesson_no: int) -> List[str]:
        files = self.get_assigned_files_for_lesson(lesson_no)
        texts = []
        for name in files:
            reading_path = next((q for q in Path(self.reading_dir).glob("**/*")
                                 if q.is_file() and q.name == name), None)
            if reading_path:
                texts.append(self.load_readings(reading_path))
            else:
                self.logger.warning(f"Assigned file not found on disk: {name}")
        return texts

    def load_lessons(self, lesson_number_or_range: Union[int, range]) -> Dict[str, List[str]]:
        lesson_range = (range(lesson_number_or_range, lesson_number_or_range + 1)
                        if isinstance(lesson_number_or_range, int)
                        else lesson_number_or_range)
        out: Dict[str, List[str]] = {}
        for lesson_no in lesson_range:
            texts = self.get_reading_texts_by_lesson(lesson_no)
            if not texts:
                self.logger.warning(f"No assigned readings found for lesson {lesson_no}.")
            out[str(lesson_no)] = texts
        return out

    def check_for_unmatched_readings(self):
        self.indexer.assign_unmatched_readings(on_update_callback=self._save_index_callback)

    # ---------------------------
    # Syllabus parsing / objectives & readings
    # ---------------------------

    def _setup_docling(self):
        from docling.backend.docling_parse_v2_backend import \
            DoclingParseV2DocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (AcceleratorDevice,
                                                        AcceleratorOptions,
                                                        PdfPipelineOptions,
                                                        TableStructureOptions)
        from docling.document_converter import (DocumentConverter,
                                                MarkdownFormatOption,
                                                PdfFormatOption,
                                                WordFormatOption)
        from docling.pipeline.base_pipeline import PipelineOptions
        from docling.pipeline.simple_pipeline import SimplePipeline

        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = False
        pdf_pipeline_options.do_table_structure = True
        pdf_pipeline_options.table_structure_options.do_cell_matching = True
        pdf_pipeline_options.ocr_options.lang = ["es"]
        pdf_pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.AUTO
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options,
                                                 backend=DoclingParseV2DocumentBackend),
            },
            allowed_formats=[InputFormat.PDF]
        )

        return converter

    def _load_or_cache_syllabus(
        self,
        syllabus_path: Union[str, Path],
        prefer_docling: bool = True,
        use_cache: bool = True,
    ) -> List[str]:
        """
        Load a DOCX/PDF/TXT syllabus into a flat list of text blocks.
        - For PDFs, prefer Docling (if available) for high-fidelity Markdown (tables, layout).
        - Fall back to MarkItDown for everything else / when Docling is unavailable.
        - Cache the converted Markdown + split-lines in the per-course index JSON and
        invalidate when the syllabus file mtime changes.
        """
        syllabus_path = Path(syllabus_path)
        if not syllabus_path.exists():
            raise FileNotFoundError(f"Syllabus not found: {syllabus_path}")

        def _split_md_into_lines(md_text: str) -> List[str]:
            # Preserve tables intact by splitting them line-by-line;
            # split paragraphs on double newlines
            blocks = md_text.replace("\r", "").split("\n\n")
            out: List[str] = []
            for b in blocks:
                b = b.strip()
                if not b:
                    continue
                if b.startswith("|"):
                    out.extend(line for line in b.split("\n") if line.strip())
                else:
                    out.append(b)
            return out

        # -------- convert (Docling if PDF + available) --------
        md_text = None
        used_converter = None

        if prefer_docling and syllabus_path.suffix.lower() == ".pdf":
            try:
                # Lazy import so your package remains optional
                self.logger.info("Converting syllabus pdf. This may take a few seconds...")
                conv = self._setup_docling()
                result = conv.convert(str(syllabus_path))
                md_text = result.document.export_to_markdown()
                used_converter = "docling"
            except Exception as e:
                self.logger.warning(f"Docling failed on {syllabus_path.name}: {e}. Falling back to MarkItDown.")

        if md_text is None:
            # MarkItDown path (handles DOCX/PDF/TXT)
            md = MarkItDown()
            converted = md.convert(str(syllabus_path))
            md_text = converted.text_content or ""
            used_converter = "markitdown"

        lines = _split_md_into_lines(md_text)

        # -------- detect tabular header indices if requested --------
        if self.tabular_syllabus:
            self.tabular_header_idx = None
            for i, line in enumerate(lines):
                lstr = line.strip().lower()
                header_keywords = ["lesson", "week", "reading", "text"]
                if self.tabular_lesson_col_name:
                    header_keywords.append(self.tabular_lesson_col_name.lower())
                if self.tabular_readings_col_name:
                    header_keywords.append(self.tabular_readings_col_name.lower())
                if lstr.startswith('|') and any(k in lstr for k in header_keywords):
                    self.tabular_header_idx = i
                    break
            if self.tabular_header_idx is not None:
                header_parts = [h.strip().lower() for h in lines[self.tabular_header_idx].split('|') if h.strip()]
                lesson_col = None
                readings_col = None
                # lesson column
                if self.tabular_lesson_col_name:
                    for idx, col in enumerate(header_parts):
                        if self.tabular_lesson_col_name.lower() in col:
                            lesson_col = idx
                if lesson_col is None:
                    for idx, col in enumerate(header_parts):
                        if any(k in col for k in ["lesson", "week"]):
                            lesson_col = idx
                            if not self.tabular_lesson_col_name:
                                self.tabular_lesson_col_name = col
                # readings column
                if self.tabular_readings_col_name:
                    for idx, col in enumerate(header_parts):
                        if self.tabular_readings_col_name.lower() in col:
                            readings_col = idx
                if readings_col is None:
                    for idx, col in enumerate(header_parts):
                        if any(k in col for k in ["reading", "text"]):
                            readings_col = idx
                            if not self.tabular_readings_col_name:
                                self.tabular_readings_col_name = col
                self.tabular_lesson_col_idx = lesson_col
                self.tabular_readings_col_idx = readings_col

        # No cache or index writing here; handled by LessonLoader
        return lines

    def extract_lesson_objectives(self, current_lesson: Union[int, str], only_current: bool = False) -> str:
        """Extract lesson objectives text for the specified lesson(s). Uses cached/in-memory syllabus lines."""
        if not hasattr(self, 'syllabus_lines') or not self.syllabus_lines:
            return "No lesson objectives provided."
        syllabus_content = self.syllabus_lines

        current_lesson = int(current_lesson)
        prev_idx, curr_idx, next_idx, end_idx = self.find_docx_indices(syllabus_content, current_lesson)

        if self.tabular_syllabus:
            prev_lesson_content = syllabus_content[prev_idx] if prev_idx is not None else ""
            curr_lesson_content = syllabus_content[curr_idx] if curr_idx is not None else ""
            next_lesson_content = syllabus_content[next_idx] if next_idx is not None else ""
        else:
            prev_lesson_content = "\n".join(syllabus_content[prev_idx:curr_idx]) if prev_idx is not None else ""
            curr_lesson_content = "\n".join(syllabus_content[curr_idx:next_idx]) if curr_idx is not None else ""
            next_lesson_content = "\n".join(syllabus_content[next_idx:end_idx]) if next_idx is not None else ""

        combined_content = "\n\n".join(filter(None, [prev_lesson_content, curr_lesson_content, next_lesson_content]))
        return curr_lesson_content if only_current else combined_content

    def find_docx_indices(self, syllabus: List[str], current_lesson: int,
                          lesson_identifier: str = "") -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Locate indices bounding previous/current/next lessons in the parsed syllabus list."""
        prev_lesson = curr_lesson = next_lesson = end_lesson = None

        if self.tabular_syllabus:
            # Table-like rows beginning with numbers or '|' (already split in load_docx_syllabus)
            lesson_pattern = re.compile(r"^(\d+)\s*\||^\|\s*(\d+)\s*\|")
            for i, line in enumerate(syllabus):
                m = lesson_pattern.match(line)
                if not m:
                    continue
                num = int((m.group(1) or m.group(2)))
                if num == current_lesson - 1:
                    prev_lesson = i
                elif num == current_lesson:
                    curr_lesson = i
                elif num == current_lesson + 1:
                    next_lesson = i
                elif num == current_lesson + 2:
                    end_lesson = i
                    break
            return prev_lesson, curr_lesson, next_lesson, end_lesson

        # Non-tabular syllabi: look for lesson/week headers, optionally prefixed with '**'
        identifiers = [lesson_identifier] if lesson_identifier else ['Lesson', 'Week', '**Lesson', '**Week']
        for ident in identifiers:
            esc = re.escape(ident)
            for i, line in enumerate(syllabus):
                if re.search(rf"{esc}\s*{current_lesson - 1}.*?:?", line):
                    prev_lesson = i
                if re.search(rf"{esc}\s*{current_lesson}.*?:", line):
                    curr_lesson = i
                if re.search(rf"{esc}\s*{current_lesson + 1}.*?:?", line):
                    next_lesson = i
                if re.search(rf"{esc}\s*{current_lesson + 2}.*?:?", line):
                    end_lesson = i
                    break
            if curr_lesson is not None:
                break
        if curr_lesson is None:
            self.logger.warning(
                f"Lesson {current_lesson} not found in syllabus. If this is a table, set `tabular_syllabus=True`.")
        return prev_lesson, curr_lesson, next_lesson, end_lesson

    # ---------------------------
    # Reading & file utilities
    # ---------------------------
    @staticmethod
    def _validate_file_path(path: Union[Path, str], name: str) -> Path:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"The {name} at path '{path}' does not exist or is not a file.")
        return path

    @staticmethod
    def _validate_dir_path(path: Union[Path, str], name: str, create_if_missing: bool = False) -> Path:
        path = Path(path) if path is not None else None
        if path is None:
            return None
        if not path.exists() and create_if_missing:
            path.mkdir(parents=True)
        elif not path.is_dir():
            raise NotADirectoryError(f"The {name} at path '{path}' does not exist or is not a directory.")
        return path

    def load_readings(self, file_path: Union[str, Path]) -> str:
        """Load text content from a single document file, with format-specific handling."""
        file_path = Path(file_path)
        text_prefix = 'title: ' + file_path.stem + "\n"
        try:
            if file_path.suffix.lower() == '.pdf':
                extracted = self.extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                try:
                    doc = Document(str(file_path))
                    extracted = "\n".join(p.text for p in doc.paragraphs)
                except PackageNotFoundError:
                    raise ValueError(f"Unable to open {file_path.name}. The file might be corrupted.")
            elif file_path.suffix.lower() == '.txt':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        extracted = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='ISO-8859-1') as f:
                        extracted = f.read()
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            if not extracted.strip():
                self.logger.warning(f"No readable text found in {file_path.name}. File may be empty or unreadable.")
        except Exception as e:
            self.logger.error(f"An error occurred while reading {file_path.name}: {e}")
            raise
        return text_prefix + extracted if extracted.strip() else "No readable text found."

    def _ocr_pdf(self, pdf_path: Union[str, Path], max_workers: int = 4, pipeline="docling") -> str:
        """Perform OCR on a PDF file and return the extracted text."""
        pdf_path = Path(pdf_path)
        return ocr_pdf(pdf_path, max_workers=max_workers)

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """Extract text content from a PDF file, with optional OCR fallback if needed."""
        text_content: List[str] = []
        pdf_path = Path(pdf_path)
        with open(str(pdf_path), 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    paragraphs = page_text.split('\n\n')
                    text_content.extend(p.strip() for p in paragraphs if p.strip())
        if not text_content and self.ocr_available():
            self.logger.warning(f"No readable text found in pdf {pdf_path.name}. Attempting OCR.")
            ocr_result = self._ocr_pdf(pdf_path, max_workers=6)
            if ocr_result.strip():
                print(f"Successful OCR of {pdf_path.name}")
                return ocr_result
            return "OCR did not produce readable text."

        elif not text_content and not self.ocr_available():
            missing = self.missing_ocr_packages()
            raise ImportError(
                f"No readable text found in PDF {pdf_path.name}.\nOCR support requires: {
                    ', '.join(missing)}.\nInstall with `pip install class_factory[ocr]`, or convert the file to readable text."
            )
        return ' '.join(text_content)

    @staticmethod
    def ocr_available() -> bool:
        """Return True if OCR dependencies are available."""
        return all([pytesseract, Image, convert_from_path])

    @staticmethod
    def missing_ocr_packages() -> List[str]:
        packages = {
            "pytesseract": pytesseract,
            "pillow": Image,
            "pdf2image": convert_from_path,
            "textblob": TextBlob,
            "docling": DocumentConverter
        }
        return [name for name, mod in packages.items() if mod is None]

    # ---------------------------
    # Beamer helpers
    # ---------------------------
    def load_beamer_presentation(self, tex_path: Path) -> str:
        tex_path = Path(tex_path)
        with open(tex_path, 'r', encoding='utf-8') as f:
            return f.read()

    def find_prior_beamer_presentation(self, lesson_no: int, max_attempts: int = 3) -> Union[Path, str]:
        for i in range(1, max_attempts + 1):
            prior = lesson_no - i
            beamer_file = (self.slide_dir / f'L{prior}.tex') if self.slide_dir else Path(f'L{prior}.tex')
            if beamer_file.is_file():
                self.logger.info(f"Found prior lesson: Lesson {prior}")
                return beamer_file
        self.logger.error(f"No prior Beamer file found within the last {max_attempts} lessons. Returning empty.")
        return "No prior presentation available."


# %%

if __name__ == "__main__":
    import os

    import yaml
    from dotenv import load_dotenv
    user_home = Path.home()
    load_dotenv()
    pdf_syllabus_path = user_home / os.getenv('pdf_syllabus_path', "")

    with open("class_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    class_name = "PS491"

    class_config = config[class_name]
    slide_dir = user_home / class_config['slideDir']
    syllabus_path = user_home / class_config['syllabus_path']
    readingsDir = user_home / class_config['reading_dir']
    is_tabular_syllabus = class_config['is_tabular_syllabus']

    lsn = 7

    loader = LessonLoader(syllabus_path=syllabus_path,
                          reading_dir=readingsDir,
                          slide_dir=slide_dir,
                          tabular_syllabus=is_tabular_syllabus,
                          class_name=class_name,
                          index_similarity=0.8)

    # %%
    loader.check_for_unmatched_readings()

    # %%
    objs = loader.extract_lesson_objectives(
        current_lesson=lsn)
    docs = loader.load_lessons(lesson_number_or_range=range(12, 14))

# %%
