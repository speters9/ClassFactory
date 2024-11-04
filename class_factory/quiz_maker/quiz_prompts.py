# quiz_prompt = """
#      You are a political scientist with expertise in {course_name}.
#      You will be creating an in-class quiz for an undergraduate-level course on {course_name}.
#      The quiz should align with the lesson objectives listed below.
#      ---
#      {objectives}
#      ---
#      Here are the texts the quiz will cover:
#      ---
#      {information}
#      ---
#      Please generate 9 questions from the readings. Questions should cover the major ideas discussed.

#      The 9 questions will consist of 3 each of the following question types:
#      - Multiple choice
#      - True/False
#      - Fill in the blank (provide plausible answer choices)

#      Return a quiz in JSON format with a variety of question types. Structure your response with keys for each question type as follows:

#     {{
#       "multiple_choice": [
#         {{
#           "question": "Question text here",
#           "A)": "Choice 1",
#           "B)": "Choice 2",
#           "C)": "Choice 3",
#           "D)": "Choice 4",
#           "correct_answer": "A, B, C, or D corresponding to the correct choice"
#         }},
#         ...
#       ],

#       "true_false": [
#         {{
#           "question": "True/False question text here",
#           "A)": "True",
#           "B)": "False",
#           "C)": "",
#           "D)": "",
#           "correct_answer": "A or B"
#         }},
#         ...
#       ],

#       "fill_in_the_blank": [
#         {{
#           "question": "Question text here",
#           "A)": "Choice 1",
#           "B)": "Choice 2",
#           "C)": "Choice 3",
#           "D)": "Choice 4",
#           "correct_answer": "Correct answer that completes the missing words"
#         }},
#         ...
#       ]
#     }}
#     ---

#     Your answer should be returned in the format specified above, with a mix of multiple choice, true/false, and fill-in-the-blank questions.
#     Every question should always contain "A)", "B)", "C)", and "D)" options, even if they are left blank.

#     ---

#     Generated questions must be different from the current list of questions, located here:

#     {prior_quiz_questions}

#     ---

#     ### IMPORTANT: Your response **must** strictly follow the JSON format above. Include only the json in your response.
#     Return only the json; don't include ```json ... ``` in your response.
#     If the JSON is invalid or extra text is included, your response will be rejected and you will not be paid.
#     """

quiz_prompt = """
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

    {additional_guidance}

    ---

    ### IMPORTANT: Responses must include only the JSON structure above.
    Return only the json; don't include ```json ... ``` in your response.
    Extra text or incorrectly formatted JSON will result in a failed task.
    """
