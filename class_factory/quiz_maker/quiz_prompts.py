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
     Please generate a total of 9 questions, consisting of:
     - 3 Multiple choice
     - 3 True/False
     - 3 Fill in the blank (include plausible answer choices)

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
    - Avoid any overlap with the current list of questions, found here:
      {prior_quiz_questions}
    - Ensure the correct answer placement is balanced across answer options.
        Each option (A, B, C, D) should be used as the correct answer at least once across the entire quiz.

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

     ### Question Requirements:
     Please generate a total of 9 questions, consisting of:
     - 3 Multiple choice
     - 3 True/False
     - 3 Fill in the blank (include plausible answer choices)

     ---

     ### Difficulty Level:
     Generate questions at a difficulty level of {difficulty_level} on a scale of 1 to 10:
     - **1-3 (Easy)**: Simple recall of definitions or basic facts.
     - **4-7 (Medium)**: Understanding relationships between concepts.
     - **8-10 (Hard)**: Application and analysis, requiring synthesis of multiple ideas.

     ---

     ### Output Format:
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

    ### Important Notes:
    - Include only JSON in your response, strictly following the format above.
    - Avoid any overlap with the current list of questions, found here:
      {prior_quiz_questions}
    - Ensure the correct answer placement is balanced across answer options.
        Each option (A, B, C, D) should be used as the correct answer at least once across the entire quiz.

    {additional_guidance}

    ---

    ### IMPORTANT: Responses **must** include only the JSON structure above.
    Return only the json. Do not include additional text or formatting (e.g., json ... ).
    Extra text or incorrectly formatted JSON will result in a failed task.
    """

quiz_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(quiz_prompt_system),
        HumanMessagePromptTemplate.from_template(quiz_prompt_human)
    ]
)
