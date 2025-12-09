
"""
quiz_prompts.py
----------------

This module defines prompt templates for generating quizzes using large language models (LLMs).
It provides both system and human message templates, as well as a composite chat prompt template,
to guide the LLM in creating quiz questions that align with lesson objectives, course content, and
specified formatting requirements.

Key Components:
- `prompt_str`: A detailed string template for quiz generation instructions.
- `quiz_prompt_system`: System message template for LLM context.
- `quiz_prompt_human`: Human message template with explicit quiz generation instructions.
- `quiz_prompt`: A `ChatPromptTemplate` combining system and human messages for use with LLM chains.

Usage:
Import `quiz_prompt` and use it as the prompt in an LLM chain for quiz question generation.
"""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

quiz_prompt_system = """You are a political scientist with expertise in {course_name}.
Your task is to create an in-class quiz for an undergraduate course that aligns with the lesson objectives.
Ensure the questions reflect an appropriate difficulty level and adhere to the specified format.
"""


quiz_prompt_human = """
## Generate a quiz according to the following guidelines

### Context and Objectives:
- The quiz should cover major ideas from the lesson objectives listed here:
{objectives}

- Here are the texts that the quiz will cover: {information}

---

### Task Instructions:
Generate **9 quiz questions** using the following breakdown:
1. **3 Multiple-choice Questions**
   - Provide 4 answer choices (A, B, C, D) for each question.
   - Ensure the correct answer is explicitly stated in the `"correct_answer"` field (e.g., "A)", "B)", "C)", or "D)").
   - Balance correct answers across all options (A, B, C, D should each be used once across the entire quiz).

2. **3 True/False Questions**
   - Format each question with "True" (A) and "False" (B) as answer options.
   - Ensure `"correct_answer"` is clearly identified as either "A" or "B".
   - "C" and "D" should return empty strings.

3. **3 Fill-in-the-blank Questions**
   - Provide the question with a blank to be filled in.
   - Include **plausible answer choices** (A, B, C, D), with the correct answer explicitly stated in the `"correct_answer"` field.


---

### Difficulty Level:
Generate questions at a difficulty level of {difficulty_level} on a scale of 1 to 10:
- **1-3 (Easy)**: Simple recall of definitions or basic facts.
- **4-7 (Medium)**: Understanding relationships between concepts.
- **8-10 (Hard)**: Application and analysis, requiring synthesis of multiple ideas.

---

### Output Format:
Your response must be a valid object matching the required schema. Do not include any markdown code block markers (such as ```json or ```).
For each question, the field "correct_answer" must be a single letter ("A", "B", "C", or "D") corresponding to the correct choice.
Do not include any example output in your response.

---


### Important Notes:
- **Correct Answer Balance**: Ensure that correct answers (A, B, C, D) are distributed evenly across all multiple-choice questions.
- **Plausible Choices**: For all questions, especially fill-in-the-blank, ensure distractor choices are realistic and relevant to the topic.
- **Avoid Duplication**: Do not include any overlap with these existing questions:
  {prior_quiz_questions}
- **Avoid author-generic verbiage**: Do not use phrases like "according to the author..." or similar, since the quiz may span multiple authors or sources. If referring to an author, you may refer to the specific authors by name.

{additional_guidance}

---

### Reminder:
- Responses **must** adhere to the defined schema.

"""

quiz_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(quiz_prompt_system),
        HumanMessagePromptTemplate.from_template(quiz_prompt_human)
    ]
)
