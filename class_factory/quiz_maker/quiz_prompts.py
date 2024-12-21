from langchain_core.messages import SystemMessage
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

prompt_str = """
     You are a political scientist with expertise in {course_name}. Your goal is to create an in-class quiz that aligns with the lesson objectives for an undergraduate course.

     **Context and Objectives:**
     The quiz should cover major ideas from the lesson objectives listed below.
     ---
     {objectives}
     ---
     Here are the texts that the quiz will cover:
     ---
     {information}
     ---

     **Question Requirements**:
     Please generate a total of **6 unique questions**, consisting of:
     - 2 Multiple choice
     - 2 True/False
     - 2 Fill in the blank (include plausible answer choices)

     Every generated question **MUST** be unique and different from all other generated questions.
     ---

     **Difficulty Level**:
     Generate questions at a difficulty level of {difficulty_level} on a scale of 1 to 10:
     - **1-3 (Easy)**: Simple recall of definitions or basic facts.
     - **4-7 (Medium)**: Understanding relationships between concepts.
     - **8-10 (Hard)**: Application and analysis, requiring synthesis of multiple ideas.

     ---

     **Output Format**:
     Return the questions in a structured JSON format as follows:

    {{
      "multiple_choice": [
        {{
          "question": "Question text here",
          "A)": "Choice 1",
          "B)": "Choice 2",
          "C)": "Choice 3",
          "D)": "Choice 4",
          "correct_answer": "A, B, C, or D corresponding to the correct choice"
        }},
        ...
      ],

      "true_false": [
        {{
          "question": "True/False question text here",
          "A)": "True",
          "B)": "False",
          "C)": "",
          "D)": "",
          "correct_answer": "A or B"
        }},
        ...
      ],

      "fill_in_the_blank": [
        {{
          "question": "Question text here",
          "A)": "Choice 1",
          "B)": "Choice 2",
          "C)": "Choice 3",
          "D)": "Choice 4",
          "correct_answer": "Correct answer that completes the missing words"
        }},
        ...
      ]
    }}

    ---

    **Important Notes**:
    - Include only JSON in your response, strictly following the format above.
    - Generate questions that are different from the current list of questions, found here:
      {prior_quiz_questions}
    - Ensure the correct answer placement is balanced across answer options.
        Each option (A, B, C, D) should be used as the correct answer at least once across the entire quiz.
    - **No Duplicate Question Generation**: Every question generated must be unique.

    {additional_guidance}

    ---

    ### IMPORTANT: Responses must include only the JSON structure above.
    Return only the json; don't include ```json ... ``` in your response.
    Extra text or incorrectly formatted JSON will result in a failed task.
    """

quiz_prompt_system = """You are a political scientist with expertise in {course_name}.
Your task is to create an in-class quiz for an undergraduate course that aligns with the lesson objectives.
Ensure the questions reflect an appropriate difficulty level and adhere to the specified format.
"""


quiz_prompt_human = """
     ### Context and Objectives:
     The quiz should cover major ideas from the lesson objectives listed below.
     ---
     {objectives}
     ---

     Here are the texts that the quiz will cover:
     ---
     {information}
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
     Here is an example output format. Your response **must** strictly follow this format:

    {{
      "multiple_choice": [
        {{
          "question": "Question text here",
          "A)": "Choice 1",
          "B)": "Choice 2",
          "C)": "Choice 3",
          "D)": "Choice 4",
          "correct_answer": "A, B, C, or D corresponding to the correct choice"
        }},
        ...
      ],

      "true_false": [
        {{
          "question": "True/False question text here",
          "A)": "True",
          "B)": "False",
          "C)": "",
          "D)": "",
          "correct_answer": "A or B"
        }},
        ...
      ],

      "fill_in_the_blank": [
        {{
          "question": "Question text here",
          "A)": "Choice 1",
          "B)": "Choice 2",
          "C)": "Choice 3",
          "D)": "Choice 4",
          "correct_answer": "Correct answer that completes the missing words"
        }},
        ...
      ]
    }}

    ---

    ### Important Notes:
    - **JSON Only**: Include only the JSON structure in your response.
    - **Correct Answer Balance**: Ensure that correct answers (A, B, C, D) are distributed evenly across all multiple-choice questions.
    - **Plausible Choices**: For all questions, especially fill-in-the-blank, ensure distractor choices are realistic and relevant to the topic.
    - **Avoid Duplication**: Do not include any overlap with these existing questions:
      {prior_quiz_questions}


    {additional_guidance}

    ---

    ### Reminder: Responses **must** adhere strictly to the JSON format provided above.
    Extra text, incorrect formatting, or invalid JSON will result in rejection.
    """

quiz_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(quiz_prompt_system),
        HumanMessagePromptTemplate.from_template(quiz_prompt_human)
    ]
)
