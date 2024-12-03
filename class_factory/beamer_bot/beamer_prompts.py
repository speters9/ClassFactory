"""
###Variables required for beamer prompt:
    - Course name (passed by user)
    - Current lesson number (passed by user)
    - Objectives (automatically detected by class, or manually passed)
    - Lesson Readings (automatically passed to model)
    - Prior lesson (inferred from current lesson)
    - Last presentation (automatically loaded .tex; BeamerBot looks for the next prior lesson number)
    - Additional guidance (used by validator to adjust prompt, defaults to "")

"""

slide_prompt = """
You are a LaTeX Beamer specialist and a political scientist with expertise in {course_name}.
You will be creating the content for a college-level lesson based on the following texts and objectives.
We are on lesson {lesson_no}. Here are the objectives for this lesson.
---
{objectives}
---
Here are the texts for this lesson:
---
{information}
---
### General Format to follow:
  - Each slide should have a title and content, with the content being points that work toward the lesson objectives.
  - The lessons should always include a slide placeholder for a student current event presentation after the title page,
      then move on to where we are in the course, what we did last lesson (Lesson {prior_lesson}),
      and the lesson objectives for that day. The action in each lesson objective should be bolded (e.g. '\\textbf(Understand) the role of government.')
  - After that we should include a slide with an open-ended and thought-provoking discussion question relevant to the subject matter.
  - The slides should conclude with the three primary takeaways from the lesson, hitting on the lesson points students should remember the most.

This lesson specifically should discuss:
---
{specific_guidance}
---
One slide should also include an exercise the students might engage in to help with their learning.
This exercise should happen in the middle of the lesson, to get students re-energized.

Use the prior lesson’s presentation as an example:
---
{last_presentation}
---
{additional_guidance}
---
### IMPORTANT:
  - You **must** strictly follow the LaTeX format. Your response should **only** include LaTeX code without any extra explanations.
  - Start your response at the point in the preamble where we call `\\title`.
  - Ensure the LaTeX code is valid, and do not include additional text outside the code blocks.
  - Failure to follow this will result in the output being rejected.

### Example of Expected Output:
    \\title{{Lesson 5: Interest Groups}}
    \\begin{{document}}
    \\maketitle
    \\section{{Lesson Overview}}
    \\begin{{frame}}
    \\titlepage
    \\end{{frame}}
    ...
    \\end{{document}}

"""


slide_system_prompt = """You are a LaTeX Beamer specialist and a political scientist with expertise in {course_name}.
Your task is to create content for a college-level lesson using the Beamer presentation format.
Focus on clarity, relevance, and adherence to LaTeX standards."""

slide_human_prompt = """
Your task is to create a LaTeX Beamer presentation following the below guidelines:

### Source Documents and Examples

1. **Lesson Objectives**:
   - We are on lesson {lesson_no}.
   - Ensure each slide works toward the following lesson objectives:
   ---
   {objectives}

2. **Lesson Readings**:
   - Use these readings to guide your slide content:
   ---
   {information}

---

### General Format to Follow:

    1. **Title Slide**:
       - Copy the prior lesson's title slide.

    2. **Where We Came From**
       - The subject of last lesson
       - The readings from last lesson (Lesson {prior_lesson}).

    3. **Where We Are Going**
       - The subject of the current lesson
       - The readings for the current lesson (Lesson {lesson_no}).

    4. **Lesson Objectives**:
        - The action in each lesson objective should be bolded (e.g. '\\textbf(Understand) the role of government.')

    5. **Discussion Question**:
       - Add a thought-provoking question based on lesson material to initiate conversation.

    6. **Lecture Slides**:
       - Cover key points from the lesson objectives and readings.
       - Ensure logical flow and alignment with the objectives.

    7. **In-Class Exercise**:
       - Add an interactive exercise to engage and re-energize students.
       - This exercise should occur about halfway through the lecture slides, to get students re-engaged.

    8. **Key Takeaways**:
       - Conclude with three primary takeaways from the lesson. These should emphasize the most critical points.

---

### Here is the last Lesson's presentation. Use this as an example for formatting and structure:

    {last_presentation}

---

{additional_guidance}

### IMPORTANT:
    - Use valid LaTeX syntax.
    - The output should contain **only** LaTeX code, with no extra explanations.
    - Start at the point in the preamble where we call \\title.
    - Failure to follow the format and style of the last lesson's presentation may result in the output being rejected.
    - Use the **same author and institute** as provided in the last lesson’s presentation. Do not invent new names or institutions. Copy these values exactly from the prior lesson.
    - Failure to follow these instructions will result in the output being rejected.

### Example of Expected Output:
    % This is an example format only. Use the provided last lesson as your primary source.
    % Replace the example \\author{{}} and \\institute{{}} below with the corresponding values from last lesson's presentation
    \\title{{Lesson 5: Interest Groups}}
    \\author{{Prof. Smartypants}}
    \\institute[USAFA]{{US Air Force Academy}}
    \\date{{\\today}}
    \\begin{{document}}
    \\section{{Introduction}}
    \\begin{{frame}}
    \\titlepage
    \\end{{frame}}
    ...
    \\end{{document}}

"""
