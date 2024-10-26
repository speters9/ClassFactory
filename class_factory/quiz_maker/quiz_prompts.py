quiz_prompt = """
     You are a political scientist with expertise in {course_name}.
     You will be creating an in-class quiz for an undergraduate-level course on {course_name}.
     The quiz should align with the lesson objectives listed below.
     ---
     {objectives}
     ---
     Here are the texts the quiz will cover:
     ---
     {information}
     ---
     Please generate 9 questions from the readings. Questions should cover the major ideas discussed.

     The 9 questions will consist of 3 each of the following question types:
     - Multiple choice
     - True/False
     - Fill in the blank (provide plausible answer choices)

     Return a quiz in JSON format with a variety of question types. Structure your response with keys for each question type as follows:

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

    Your answer should be returned in the format specified above, with a mix of multiple choice, true/false, and fill-in-the-blank questions.
    Every question should always contain "A)", "B)", "C)", and "D)" options, even if they are left blank.

    ---

    Generated questions must be different from the current list of questions, located here:

    {prior_quiz_questions}

    ---

    ### IMPORTANT: Your response **must** strictly follow the JSON format above. Include only the json in your response.
    Return only the json; don't include ```json ... ``` in your response.
    If the JSON is invalid or extra text is included, your response will be rejected and you will not be paid.
    """
