"""
pptx_prompts.py
---------------

This module defines prompt templates for generating PowerPoint presentations using large language models (LLMs).
It provides both system and human message templates to guide the LLM in creating structured,
pedagogically sound slides for college-level lessons.

Key Components:
- `pptx_system_prompt`: System message template for LLM context (course, standards).
- `pptx_human_prompt`: Human message template with explicit slide structure and lesson requirements.

Usage:
Import `pptx_system_prompt` and `pptx_human_prompt` for use in LLM chains or prompt construction in PptxSlideGenerator.

Note:
These prompts DO NOT use LaTeX syntax. Content should be plain text with markdown-like formatting.
"""

pptx_system_prompt = """You are a presentation design specialist and a political scientist with expertise in {course_name}.
        Your task is to create content for a college-level lesson using PowerPoint presentation format.
        Focus on clarity, visual appeal, conciseness, and pedagogical effectiveness."""


pptx_human_prompt = """
## Create a structured outline for a PowerPoint presentation, following these guidelines:

### Source Documents and Examples
1. **Lesson Objectives**:
   - We are on lesson {lesson_no}.
   - Ensure each slide works toward the following lesson objectives:
   {objectives}

2. **Lesson Readings**:
   - Use these readings to guide your slide content:
   {information}

---

### Output Format (IMPORTANT):
Output a presentation object with metadata and slides according to the required schema.

Top-level fields:
- "title": Lesson title (e.g., "Lesson {lesson_no}: Democratic Systems")
- "author": Author name (extract from prior presentation)
- "institute": Institution name (extract from prior presentation)
- "slides": List of slide objects

Note: The date field will be auto-filled with the current date during slide generation.

Each slide object must have:
- "title": the slide title (string)
- "content": the slide content as plain text with markdown-like formatting (string)
- "slide_type": a label for the slide type (e.g., "title", "objectives", "content", "summary")
- "bullet_points": (optional) list of bullet points for the slide
- "notes": (optional) speaker notes for the slide

Example:
{{
  "title": "Lesson {lesson_no}: Democratic Systems",
  "author": "Dr. Smith",
  "institute": "University of Political Science",
  "slides": [
    {{
      "title": "Lesson {lesson_no}: Democratic Systems",
      "content": "",
      "slide_type": "title"
    }},
    {{
      "title": "Lesson Objectives",
      "content": "",
      "bullet_points": ["**Understand** the role of government", "**Analyze** key political events"],
      "slide_type": "objectives",
      "notes": "Emphasize the action verbs in each objective"
    }},
    {{
      "title": "Key Concepts",
      "content": "Democracy is a form of government...",
      "bullet_points": ["Popular sovereignty", "Rule of law", "Individual rights"],
      "slide_type": "content"
    }}
  ]
}}

---

### Content Formatting Guidelines (CRITICAL):
- **BREVITY IS ESSENTIAL**: All text must fit on a single slide without scrolling
- Use **bold** for emphasis (NOT LaTeX commands like \textbf{{}})
- Use *italics* for terms (NOT LaTeX commands like \textit{{}})
- **IMPORTANT**: When you have a LIST of items, ALWAYS use the "bullet_points" array field
- NEVER put lists in the "content" field - use "bullet_points" instead
- The "content" field is ONLY for paragraph text or single statements
- **Maximum 5 bullet points per slide** (fewer is better)
- **Each bullet point should be 1-2 lines maximum** (about 10-15 words)
- Use short phrases in bullets, NOT full sentences
- NO LaTeX syntax or commands
- Content should be presentation-ready text that FITS on a slide

---

### Slide Sequence to Include:
1. Title Slide: Use the current lesson title (e.g., "Lesson {lesson_no}: <lesson_title>") with slide_type "title"
2. Where We Are in the Course: Summarize last lesson and current lesson readings
3. Lesson Objectives: Use bullet_points with **bold** action verbs (e.g., '**Understand** the role of government')
4. Discussion Question: Add a thought-provoking question based on lesson material
5. Content Slides: Cover key points from objectives and readings; ensure logical flow
   - Break complex topics into multiple slides (one main idea per slide)
   - Use bullet_points for lists
   - Keep text concise and readable
6. In-Class Exercise: Add an interactive exercise about halfway through
7. Key Takeaways: Three main points with **bold** or *italic* formatting for key terms
8. Next Time: Title and readings for the next lesson
9. References: List key readings and sources

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
- Do NOT use LaTeX syntax or commands anywhere in the content.
- Use plain text with markdown-like formatting (**bold**, *italic*).
- Use the bullet_points field for lists, not itemize environments.
- Keep slides concise and visually clean.
- Failure to follow this format will result in the output being rejected.
"""
