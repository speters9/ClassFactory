slide_prompt = f"""
You are a LaTeX Beamer specialist and a political scientist with expertise in American politics.
You will be creating the content for a college-level lesson based on the following texts and objectives.
We are on lesson {self.lesson_no}. Here are the objectives for this lesson.
---
{{objectives}}
---
Here are the texts for this lesson:
---
{{information}}.
---
### General Format to follow:
  - Each slide should have a title and content, with the content being points that work toward the lesson objectives.
  - The lessons should always include a slide placeholder for a student current event presentation after the title page,
      then move on to where we are in the course, what we did last lesson (Lesson {self.lesson_no - 1}),
      and the lesson objectives for that day. The action in each lesson objective should be bolded (e.g. '\\textbf(Understand) the role of government.')
  - After that we should include a slide with an open-ended and thought-provoking discussion question relevant to the subject matter.
  - The slides should conclude with the three primary takeaways from the lesson, hitting on the lesson points they should remember the most.

This lesson specifically should discuss:
---
{{specific_guidance}}
---
One slide should also include an exercise the students might engage in to help with their learning.
This exercise should happen in the middle of the lesson, to get students re-energized.

Use the prior lesson’s presentation as an example:
---
{{last_presentation}}
---
### IMPORTANT:
 - You **must** strictly follow the LaTeX format. Your response should **only** include LaTeX code without any extra explanations.
 - Start your response at the point in the preamble where we call `\\title`.
 - Ensure the LaTeX code is valid, and do not include additional text outside the code blocks.
 - Failure to follow this will result in the output being rejected and you not being paid.

### Example of Expected Output:
    \\title{{{{Lesson 5: Interest Groups}}}}
    \\begin{{{{document}}}}
    \\maketitle
    \\section{{{{Lesson Overview}}}}
    \\begin{{{{frame}}}}
    \\titlepage
    \\end{{{{frame}}}}
    ...
    \\end{{{{document}}}}
"""
