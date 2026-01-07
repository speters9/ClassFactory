"""
latex_prompts.py
-----------------

This module defines prompt templates for generating LaTeX Beamer presentations using large language models (LLMs).
It provides both system and human message templates, as well as a composite prompt, to guide the LLM in creating structured, pedagogically sound slides for college-level lessons.

Key Components:
- `latex_system_prompt`: System message template for LLM context (course, standards).
- `latex_human_prompt`: Human message template with explicit slide structure and lesson requirements.

Usage:
Import `latex_system_prompt` and `latex_human_prompt` for use in LLM chains or prompt construction in LatexSlideGenerator.
"""
latex_system_prompt = """You are a LaTeX Beamer specialist and a political scientist with expertise in {course_name}.
        Your task is to create content for a college-level lesson using the Beamer presentation format.
        Focus on clarity, relevance, and adherence to LaTeX standards."""


latex_human_prompt = """
## Create a structured outline for a LaTeX Beamer presentation, following these guidelines:

### Source Documents and Examples
1. **Lesson Objectives (AUTHORITATIVE SOURCE)**:
   - We are on lesson {lesson_no}.
   - The following lesson objectives are the authoritative objectives for this lesson
     (extracted from the course syllabus or provided by the instructor):
   {objectives}

   **CRITICAL**: These are the ONLY objectives you should use. Do NOT use any "learning objectives"
   or "chapter objectives" that may appear in the textbook readings below. If the readings contain
   objectives, IGNORE them completely. Only use the objectives provided above.

2. **Lesson Readings**:
   - Use these readings to guide your slide content:
   {information}

   **Note**: Some readings (especially textbook chapters) may include their own learning objectives.
   You must IGNORE those and only use the syllabus objectives provided in section 1 above.

---

### Output Format (IMPORTANT):
Output a list of slides according to the required schema. Each slide must be an object with:
- "title": the slide title (string)
- "content": the LaTeX content for the slide body (string)
- "slide_type": (optional) a label for the slide type (e.g., "objectives", "summary", "titlepage)

Example:
[
  {{"title": "Lesson Objectives", "content": "\\begin{{itemize}} ... \\end{{itemize}}", "slide_type": "objectives"}},
  {{"title": "Key Takeaways", "content": "...", "slide_type": "summary"}}
]

---

### Slide Sequence to Include:
1. Title Slide: Use the current lesson title (e.g., "Lesson {lesson_no}: <lesson_title>").
2. Where We Are in the Course: Summarize last lesson and current lesson readings.
3. Lesson Objectives: Bold the action in each objective (e.g., '\\textbf{{Understand}} the role of government').
4. Discussion Question: Add a thought-provoking question based on lesson material.
5. Lecture Slides: Cover key points from objectives and readings; ensure logical flow. Ensure a logical ordering of slides.
6. In-Class Exercise: Add an interactive exercise about halfway through.
7. Key Takeaways: Three main points, bold/italicize key terms.
8. Next Time: Title and readings for the next lesson.
9. References: Only one thing: %\\printbibliography

---

### Specific guidance for this lesson:
{specific_guidance}

---
{additional_guidance}

---

### Example of previous presentation:
- Use the presentation from last lesson as an example for formatting and structure:
{last_presentation}

---

### IMPORTANT:
- Output only the schema-compliant array of slides, with no extra explanations or formatting.
- Do NOT output a full LaTeX document. Only provide the structured slide data as described above.
- Failure to follow this format will result in the output being rejected.
- Remember, this is for a Beamer presentation, so all text and fonts should be in LaTeX format.
- When generating the lecture slides, make sure to synthesize the readings and their key concepts into the broader theme of the lesson; don't just summarize the readings.
"""


# Backward compatibility aliases
beamer_system_prompt = latex_system_prompt
beamer_human_prompt = latex_human_prompt
