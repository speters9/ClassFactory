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
beamer_system_prompt = """You are a LaTeX Beamer specialist and a political scientist with expertise in {course_name}.
        Your task is to create content for a college-level lesson using the Beamer presentation format.
        Focus on clarity, relevance, and adherence to LaTeX standards."""

beamer_human_prompt = """
   ## Create a LaTeX Beamer presentation following the below guidelines:

   ### Source Documents and Examples

   1. **Lesson Objectives**:
      - We are on lesson {lesson_no}.
      - Ensure each slide works toward the following lesson objectives:
      {objectives}

   2. **Lesson Readings**:
      - Use these readings to guide your slide content:
      {information}

   ---

   ### General Format to Follow:

   1. **Title Slide**:
      - Copy the prior lesson's title slide, **leave author and institution blank**.

   2. **Where We Are in the Course**
      - Last time: <Title of last lesson>
         - The readings from last lesson (Lesson {prior_lesson}).
         - **Include every assigned reading from this lesson.**
      - Today: <Title of the current lesson>
         - The readings for the current lesson (Lesson {lesson_no}).
         - **Include every assigned reading from this lesson**

   3. **Lesson Objectives**:
         - The action in each lesson objective should be bolded (e.g. '\\textbf(Understand) the role of government.')

   4. **Discussion Question**:
      - Add a thought-provoking question based on lesson material to initiate conversation.

   5. **Lecture Slides**:
      - Cover key points from the lesson objectives and readings.
      - Ensure logical flow and alignment with the objectives.

   6. **In-Class Exercise**:
      - Add an interactive exercise to engage and re-energize students.
      - This exercise should occur about halfway through the lecture slides, to get students re-engaged.

   7. **Key Takeaways**:
      - Conclude with three primary takeaways from the lesson. These should emphasize the most critical points.
      - Bold or italicize the key terms for emphasis.

   8. **Next Time**:
      - Provide the title of the next lesson.
      - Include the assigned readings for the next lesson.

   9. **References**:
      - This slide should have only one thing on it:
      - %\\printbibliography
   ---

   ### Specific guidance for this lesson:

   {specific_guidance}

   ---

   ### Example of Expected Output:
         % This is an example format only. Use the provided last lesson as your primary source.
         % Note that the author and institute variables are left blank.
         \\title{{Lesson 5: Interest Groups}}
         \\author{{}}
         \\institute[]{{}}
         \\date{{\\today}}

         \\begin{{document}}

         \\begin{{frame}}
         \\titlepage
         \\end{{frame}}
         \\section{{Introduction}}
         ...
         \\end{{document}}


   {additional_guidance}

   ---

   ### Example of previous presentation:
   - Use the presentation from last lesson as an example for formatting and structure:
   {last_presentation}

   ---

   ### IMPORTANT:
   - Use valid LaTeX syntax.
   - The output should contain **only** LaTeX code, with no extra explanations.
   - Start at the point in the preamble where we call \\title.
   - Failure to follow the format and style of the last lesson's presentation may result in the output being rejected.
   - Use the **same author and institute** as provided in the last lesson’s presentation. Do not invent new names or institutions. Copy these values exactly from the prior lesson.
   - If unable to identify the author and institute from the last lesson, just leave them blank.
   - Failure to follow these instructions will result in the output being rejected.
   """
