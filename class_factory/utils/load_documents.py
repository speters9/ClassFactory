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
    - load_docx_syllabus: Load and parse DOCX syllabus content
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
# %%
import json
import logging
import re
import string
import time
import traceback
import unicodedata
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
    def __init__(self, reading_dir: Union[Path, str], syllabus: List[str], index_path: Union[Path, str],
                 tabular_syllabus: bool = False, index_similarity: float = 0.8):
        self.reading_dir = Path(reading_dir)
        self.syllabus = syllabus
        self.tabular_syllabus = tabular_syllabus
        self.index_path = Path(index_path)
        self.index_similarity = index_similarity
        self.logger = logger_setup(logger_name="reading_indexer", log_level=logging.INFO)

        self.reading_index = self._load_or_build_reading_index()  # Load or build the reading index on initialization

    def _load_or_build_reading_index(self) -> dict:  # This method is now a stub for compatibility; always builds and returns the lesson-centric index
        if self.index_path.exists():
            self.logger.info(f"Loading lesson-centric index from {self.index_path}")

            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        self.logger.info("No reading index found. Building a new one…")
        index = self.build_lesson_centric_index(save_path=self.index_path)

        def handle_user_choices(user_choices, lesson_index):  # This callback can be customized to update the lesson-centric index if needed
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)  # After building, prompt user for ambiguous/low-confidence matches only

        try:
            prompt_user_for_reading_matches(index,
                                            match_threshold=self.index_similarity,
                                            top_n=3,
                                            on_submit_callback=handle_user_choices)
        except ImportError:
            self.logger.warning("ipywidgets not available; skipping interactive reading assignment.")
        except Exception as e:
            self.logger.warning(f"Error during interactive reading assignment: {e}")  # Save initial index (before user input)
            traceback.print_exc()

        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

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
        STRICT mode for linear syllabi:
        Within each lesson's 'Readings:' section, readings must be separated by
        - bullets at the start of a line (*, -, •, etc.), or
        - semicolons (;).
        """
        lines: List[str] = syllabus_lines
        readings_by_lesson: Dict[int, List[dict]] = {}

        BULLET_LINE = re.compile(r'(?m)^\s*([*\-•·▪])\s+')
        SEMIS = re.compile(r'\s*;\s*')
        lesson_hdr = re.compile(r'^\s*(?:Lesson|Week)\s*([0-9]{1,3})\b', flags=re.I)

        def _clean(s: str) -> str:
            s = s.replace('|', ' ')
            s = re.sub(r'\s{2,}', ' ', s)
            s = re.sub(r'(\w)-\s+(\w)', r'\1\2', s)
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

            return {"raw": original, "authors": None, "year": None,
                    "title": item.strip('"“”'), "work": None, "chapter": None, "pages": pages, "assigned": True}

        # find lesson block boundaries
        starts: List[Tuple[int, int]] = []
        for i, raw in enumerate(lines):
            s = _clean(raw)
            m = lesson_hdr.match(s)
            if m:
                starts.append((i, int(m.group(1))))
        starts.sort(key=lambda t: t[0])
        bounds: List[Tuple[int, int, int]] = []
        for idx, (si, num) in enumerate(starts):
            ei = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
            bounds.append((num, si, ei))

        def _strict_split_block(block_lines: List[str]) -> List[str]:
            # Join block, force bullets to newline, then split by semicolons and newlines.
            text = "\n".join(block_lines)
            text = BULLET_LINE.sub('\n', text)
            text = re.sub(r'\s\*\s+(?=\S)', '\n', text)  # inline " * " → newline
            items: List[str] = []
            for raw in text.split('\n'):
                raw = raw.strip()
                if not raw:
                    continue
                items.extend(p.strip() for p in SEMIS.split(raw) if p.strip())
            return [_clean(it) for it in items if len(re.sub(r'\s+', '', it)) >= 2]

        # iterate lessons
        for lesson_num, si, ei in bounds:
            block = [lines[si+i] for i in range(ei - si)]

            # Prefer explicit "Readings:" header
            rs = re_i = None
            for j, s in enumerate(block):
                if re.match(r'^\s*Readings?\s*:\s*$', _clean(s), flags=re.I):
                    rs = j + 1
                    break

            if rs is not None:
                for k in range(rs, len(block)):
                    if re.match(r'^\s*(Objectives|Learning\s+Objectives|Topics|Agenda|Activities|Assignments|Due|Quiz|Exam|VAT|Notes|Discussion)\b',
                                _clean(block[k]), flags=re.I):
                        re_i = k
                        break
                if re_i is None:
                    re_i = len(block)
                reading_lines = [_clean(s) for s in block[rs:re_i]]
            else:
                # fallback: take whole lesson block (strict splitting will only honor bullets/semicolons)
                reading_lines = [_clean(s) for s in block]

            items = _strict_split_block(reading_lines)
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
        for lesson_key, lesson_entry in self.reading_index.items():
            lesson = lesson_entry.get("number", lesson_key)
            readings = []
            for reading, entry in lesson_entry.get("readings", {}).items():
                assigned_file = entry.get("assigned_file")
                if assigned_file:
                    readings.append(assigned_file)
            seen = set()
            deduped = []
            for r in readings:
                if r not in seen:
                    deduped.append(r)
                    seen.add(r)
            simple_index[str(lesson)] = deduped
        return simple_index

    def export_lesson_readings_index(self, output_path=None):
        simple_index = self.get_lesson_readings_map()
        idx = dict(self.reading_index)
        idx.pop("lesson_readings_index", None)
        new_idx = {"lesson_readings_index": simple_index}
        new_idx.update(idx)
        if output_path is None:
            output_path = self.index_path
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_idx, f, indent=2)
        print(f"Simplified lesson-to-readings index saved as 'lesson_readings_index' at the top of {output_path}")

    def build_lesson_centric_index(
        self,
        save_path: Optional[Path] = None,
        top_n: int = 3,
        similarity_method: str = 'max',  # or 'tfidf' or 'rf', 'ensemble'
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

        # Pass loaded syllabus and tabular flag
        lesson_readings = self.extract_lesson_readings_from_syllabus()
        index = {}
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
                    else:
                        # explicit 'max' option if you want it
                        combined = max(rf_score, tf_score)

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
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
        self.reading_index = index
        return index

    def assign_unmatched_readings(self):

        def handle_assignments(assignments, updated_index=None):
            idx = updated_index if updated_index is not None else self.reading_index
            for lesson_key, lesson_entry in idx.items():
                lesson = lesson_entry.get("number", lesson_key)
                assigned_files = set(assignments.get(lesson, []))
                for reading, entry in lesson_entry.get("readings", {}).items():
                    matches = entry.get("matches", [])
                    assigned_file = None
                    for m in matches:
                        if m["file"] in assigned_files:
                            assigned_file = m["file"]
                            break
                    if assigned_file is None:
                        if entry.get("assigned_file") in assigned_files:
                            assigned_file = entry.get("assigned_file")
                        elif reading in assigned_files:
                            assigned_file = reading
                    entry["assigned_file"] = assigned_file
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(idx, f, indent=2)
            print("\nAssignments updated and saved.")
            self.export_lesson_readings_index()

        try:
            lesson_readings = self.extract_lesson_readings_from_syllabus()
            expected_counts = {lesson: len(readings) for lesson, readings in lesson_readings.items()}
            lessons = sorted(expected_counts.keys())
            current_assignments = {lesson: [] for lesson in lessons}
            all_readings = set()
            for lesson in lessons:
                lesson_key = str(lesson)
                lesson_entry = self.reading_index.get(lesson_key)
                if not lesson_entry:
                    continue
                for reading, entry in lesson_entry["readings"].items():
                    assigned_file = entry.get("assigned_file")
                    if assigned_file:
                        current_assignments[lesson].append(assigned_file)
                        all_readings.add(assigned_file)
                    for m in entry.get("matches", []):
                        all_readings.add(m["file"])
            all_readings = list(all_readings)
            assigned_readings = set()
            for files in current_assignments.values():
                assigned_readings.update(files)
            unassigned = sorted(set(all_readings) - assigned_readings)
            if not unassigned:
                print("All readings are assigned. Saving lesson_readings_index as is.")
                self.export_lesson_readings_index()
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
        self.indexer = ReadingIndexer(index_path=self.index_path,
                                      reading_dir=self.reading_dir,
                                      syllabus=self.load_docx_syllabus(self.syllabus_path),
                                      tabular_syllabus=self.tabular_syllabus,
                                      index_similarity=self.index_similarity
                                      )
        self.reading_index: Dict[str, Dict] = self.indexer.reading_index

    def get_assigned_files_for_lesson(self, lesson_no: int) -> List[str]:
        """Return assigned filenames for the given lesson (new index shape)."""
        lesson_entry = self.reading_index.get(str(lesson_no)) or self.reading_index.get(lesson_no)
        if not lesson_entry:
            return []
        out = []
        for _raw, r in lesson_entry.get("readings", {}).items():
            f = r.get("assigned_file")
            if f:
                out.append(f)
        # stable + dedup
        seen = set()
        dedup = []
        for f in out:
            if f not in seen:
                dedup.append(f)
                seen.add(f)
        return dedup

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
        self.indexer.assign_unmatched_readings()

    # ---------------------------
    # Syllabus parsing / objectives & readings
    # ---------------------------

    def load_docx_syllabus(self, syllabus_path: Union[str, Path]) -> List[str]:
        """Load a DOCX (or convertible) syllabus into a flat list of text blocks, splitting table rows.
        For tabular syllabi, also detect and store lesson/reading column indices."""
        max_retries = 3
        retry_delay = 10
        syllabus_path = Path(syllabus_path)
        md = MarkItDown()
        for attempt in range(max_retries):
            try:
                raw_content = md.convert(str(syllabus_path))
                lines = raw_content.text_content.split("\n\n")
                # If tables present, split their rows so matching works for tabular syllabi
                expanded: List[str] = []
                for line in lines:
                    if line.startswith("|"):
                        expanded.extend(line.split("\n"))
                    else:
                        expanded.append(line)
                # Detect and store tabular column indices if needed
                if self.tabular_syllabus:
                    self.tabular_header_idx = None
                    for i, line in enumerate(expanded):
                        lstr = line.strip().lower()
                        # Check for user-specified lesson/reading column names if provided
                        header_keywords = ["lesson", "week", "reading", "text"]
                        if self.tabular_lesson_col_name:
                            header_keywords.append(self.tabular_lesson_col_name.lower())
                        if self.tabular_readings_col_name:
                            header_keywords.append(self.tabular_readings_col_name.lower())
                        if lstr.startswith('|') and any(k in lstr for k in header_keywords):
                            self.tabular_header_idx = i
                            break
                    if self.tabular_header_idx is not None:
                        header_parts = [h.strip().lower() for h in expanded[self.tabular_header_idx].split('|') if h.strip()]
                        lesson_col = None
                        readings_col = None
                        # Find lesson column
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
                        # Find readings column
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
                return expanded
            except (PackageNotFoundError, FileConversionException, UnsupportedFormatException, PermissionError):
                if attempt < max_retries - 1:
                    print(f"Document `{syllabus_path.name}` is open. Retrying in {retry_delay} seconds…")
                    time.sleep(retry_delay)
                else:
                    raise PackageNotFoundError("Unable to open the document after multiple attempts. Please close it and try again.")

    def extract_lesson_objectives(self, current_lesson: Union[int, str], only_current: bool = False) -> str:
        """Extract lesson objectives text for the specified lesson(s). Supports tabular syllabi via indices search."""
        if not self.syllabus_path:
            return "No lesson objectives provided."
        syllabus_path = Path(self.syllabus_path)
        syllabus_content = self.load_docx_syllabus(syllabus_path)

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

    class_name = "PS460"

    class_config = config[class_name]
    slide_dir = user_home / class_config['slideDir']
    syllabus_path = user_home / class_config['syllabus_path']
    readingsDir = user_home / class_config['reading_dir']
    is_tabular_syllabus = class_config['is_tabular_syllabus']

    lsn = 8

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
