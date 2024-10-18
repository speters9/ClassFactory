"""
This script creates an interactive quiz interface using Gradio based on quiz data from an Excel file.
It takes the results of quiz generation (e.g., from 01_make_a_quiz.py) and transforms them into a fully interactive
web-based quiz, allowing users to navigate through the questions, submit answers, and receive feedback.

Workflow:
1. **Load Quiz Data**: Reads quiz questions and answers from an Excel file with a comparable structure, including
   multiple-choice questions, answer options (A, B, C, D), and the correct answer.
2. **Interactive Interface**:
   - Displays one question at a time along with multiple-choice options.
   - Users can submit their answers and receive immediate feedback (correct/incorrect).
   - "Next" and "Back" buttons allow users to navigate between questions.
3. **Feedback and Navigation**: For each question, feedback is provided based on the user's input, and users can navigate
   forward or backward through the quiz.
4. **Customization**: The interface supports custom themes and CSS styling for visual consistency and enhanced user experience.
5. **Saving and Launching**: The Gradio app is launched with options to share, making it accessible for students or participants.

Dependencies:
- Requires `Gradio` for the interactive interface and `pandas` for loading quiz data from an Excel file.
- The quiz data file should be structured with columns for 'question', 'A)', 'B)', 'C)', 'D)', and 'correct_answer'.
- Ensure the necessary environment variables (e.g., `syllabus_path`) are set correctly for locating the input quiz file.

This script generates a fully functional quiz web app, ready for real-time user interaction and feedback.
"""

import csv
# base libraries
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Union

import gradio as gr
import pandas as pd
import qrcode
# env setup
from dotenv import load_dotenv
from pyprojroot.here import here

wd = here()
load_dotenv()
pd.set_option('display.max_columns', 10)

# Path definitions
syllabus_path = Path(os.getenv('syllabus_path'))
inputDir = wd / "data/processed"


# %%

# Load CSV data


def load_data(quiz_path: Path, sample: int = None) -> pd.DataFrame:
    """
    Loads quiz data from an Excel file.

    Args:
        quiz_path (Path): Path to the Excel file containing quiz questions.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the quiz data, with columns for questions,
                      choices (A, B, C, D), and correct answers.
    """
    quiz = pd.read_excel(quiz_path)
    if not sample:
        return quiz
    else:
        return quiz.sample(sample)

# Function to log the quiz results


def log_quiz_result(user_id: str, question: str, user_answer: str, correct_answer: str, is_correct: bool, output_dir: Union[Path, str] = None):
    """
    Logs the quiz result to a CSV file.

    Args:
        user_id (str): The unique identifier for the user (can be a session ID or timestamp).
        question (str): The quiz question.
        user_answer (str): The user's submitted answer.
        correct_answer (str): The correct answer for the question.
        is_correct (bool): Whether the user's answer was correct.

    Raises: KeyError: if no output directory specified
    """
    # Reformat the datetime to year-mon-dateThr-min-sec
    formatted_datetime = datetime.now().strftime('%Y-%m-%d')

    if output_dir:
        save_dir = Path(output_dir) / "quiz_results"
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise KeyError("If saving quiz outputs, an output directory must be specified")

    # Filepath for the quiz results CSV
    csv_file_path = save_dir / f'{user_id}_{formatted_datetime}.csv'

    # Check if the file already exists to avoid overwriting the header
    file_exists = csv_file_path.exists()

    # Log the result to the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file doesn't exist
        if not file_exists:
            writer.writerow(['user_id', 'question', 'user_answer', 'correct_answer', 'is_correct', 'timestamp'])

        # Append the user's quiz result as a new row
        writer.writerow([user_id, question, user_answer, correct_answer, is_correct, datetime.now().isoformat()])


def submit_answer(current_index: int, user_answer: str, quiz_data: pd.DataFrame, user_id: str, save_results: bool, output_dir: Union[Path, str] = None) -> str:
    """
    Checks the submitted answer for the current quiz question and logs the result if log is True.

    Args:
        current_index (int): The index of the current question.
        user_answer (str): The user's selected answer for the current question.
        quiz_data (pd.DataFrame): The DataFrame containing quiz questions and answers.
        user_id (str): A unique identifier for the user (can be a session ID or timestamp).
        log (bool): Whether to log the quiz result.

    Returns:
        str: Feedback indicating whether the user's answer is correct or incorrect,
             along with the correct answer if incorrect.
    """
    row = quiz_data.iloc[current_index]
    correct_answer_key = row['correct_answer']  # Get 'A', 'B', 'C', or 'D'
    correct_answer_text = row[f"{correct_answer_key})"]

    # Check if the answer is correct
    is_correct = user_answer == correct_answer_text
    feedback = f"Question {current_index + 1}: {'Correct!' if is_correct else f'Incorrect. The correct answer was: {correct_answer_text}.'}"

    # Log the result only if the log flag is True
    if save_results:
        log_quiz_result(user_id, row['question'], user_answer, correct_answer_text, is_correct, output_dir)

    return feedback


def next_question(current_index: int, quiz_data: pd.DataFrame) -> tuple:
    """
    Advances to the next quiz question, updating the question and choices.

    Args:
        current_index (int): The index of the current question.
        quiz_data (pd.DataFrame): The DataFrame containing quiz questions and answers.
    Returns:
        tuple: A tuple containing:
               - gr.Radio.update: The updated question and choices to display.
               - str: Empty feedback text.
               - int: The updated current question index.
               If the last question is reached, returns a "Quiz completed" message and hides the quiz interface.
    """
    current_index += 1

    # If there are more questions, update the question display
    if current_index < len(quiz_data):
        row = quiz_data.iloc[current_index]
        choices = [row['A)'], row['B)'], row['C)'], row['D)']]
        choices = [choice for choice in choices if pd.notna(choice) and choice != ""]
        return gr.update(label=f"Question {current_index + 1}: {row['question']}", choices=choices), "", current_index

    # If there are no more questions, show the quiz is completed
    return gr.update(visible=False), "Quiz completed!", current_index

# Function to move to the previous question


def prev_question(current_index: int, quiz_data: pd.DataFrame) -> tuple:
    """
    Returns to the previous quiz question, updating the question and choices.

    Args:
        current_index (int): The index of the current question.
        quiz_data (pd.DataFrame): The DataFrame containing quiz questions and answers.
    Returns:
        tuple: A tuple containing:
               - gr.Radio.update: The updated question and choices to display.
               - str: Empty feedback text.
               - int: The updated current question index, ensuring the index doesn't go below 0.
    """
    current_index -= 1
    if current_index < 0:
        current_index = 0  # Prevent going before the first question

    row = quiz_data.iloc[current_index]
    choices = [row['A)'], row['B)'], row['C)'], row['D)']]
    choices = [choice for choice in choices if pd.notna(choice) and choice != ""]

    # Update to the previous question
    return gr.update(label=f"Question {current_index + 1}: {row['question']}", choices=choices), "", current_index


theme = gr.themes.Soft(
    text_size="lg",
    spacing_size="lg",
)

css = """
    body {
        display: flex;
        justify-content: center;   /* Center horizontally */
        align-items: center;       /* Center vertically */
        height: 100vh;             /* Full viewport height */
        margin: 0;                 /* Remove default margins */
    }
    gradio-app {
        max-width: 800px;          /* Limit width of the app */
        width: 100%;               /* Make sure it doesn't exceed the set width */
    }
    .feedback-box {
        min-height: 50px;          /* Set a minimum height for the feedback box */
        transition: all ease-in-out; /* Smooth transition for content changes */
    }
    .gradio-loading {
        display: none;  /* Completely hide the loading spinner */
    }
    """


def quiz_app(quiz_data: pd.DataFrame, save_results: bool = True, output_dir: Union[Path, str] = None) -> None:
    """
    Launches an interactive quiz application using Gradio.

    Args:
        quiz_data (pd.DataFrame): The quiz questions and answers.
        log (bool): Whether to log quiz results (default is True).
    """
    # Gradio Interface
    with gr.Blocks(theme=theme, css=css) as iface:
        # Add a title to the quiz
        gr.Markdown("### PS211 Review Quiz")

        # Generate a unique user identifier (this stays unique per user session)
        user_id = str(uuid.uuid4())[:8]  # Create unique user ID

        # State to track current question index and quiz data
        current_index = gr.State(value=0)
        quiz_state = gr.State(value=quiz_data)

        # Display for question and feedback
        question_display = gr.Radio(choices=[], label="", elem_classes="custom-question")
        feedback_display = gr.Textbox(value="Awaiting submission...", label="Feedback",
                                      interactive=False, elem_classes="feedback-box")

        # Create a row to hold the Submit and Next buttons
        with gr.Row():
            back_button = gr.Button("Back")
            submit_button = gr.Button("Submit")
            next_button = gr.Button("Next")

        # Logic to show feedback after submission (pass user_state to submit_answer)
        submit_button.click(
            submit_answer,
            inputs=[current_index, question_display, quiz_state, gr.State(
                user_id), gr.State(save_results), gr.State(output_dir)],  # Pass the log flag
            outputs=[feedback_display]
        )

        # Logic to move to the next question and clear feedback
        next_button.click(
            next_question,
            inputs=[current_index, quiz_state],
            outputs=[question_display, feedback_display, current_index]
        )

        # Logic to move to the previous question and clear feedback
        back_button.click(
            prev_question,
            inputs=[current_index, quiz_state],
            outputs=[question_display, feedback_display, current_index]
        )

        # Initialize the first question at startup
        iface.load(fn=next_question, inputs=[gr.State(-1), quiz_state],
                   outputs=[question_display, feedback_display, current_index],
                   show_progress='hidden')

        iface.launch(share=True,
                     prevent_thread_lock=True)

        url = iface.share_url
        if url:
            # Generate QR code from the Gradio URL
            qr = qrcode.make(url)
            # Reformat the datetime to year-mon-dateThr-min-sec
            formatted_datetime = datetime.now().strftime('%Y-%m-%d')

            qr_path = Path(output_dir) / f"quiz_results/gradio_qr_code_{formatted_datetime}.png"
            qr.save(qr_path)

            # print(f"Gradio URL: {url}")
            print(f"\nQR code saved as {str(qr_path)}")
        else:
            print("Could not generate a shareable URL.")


if __name__ == "__main__":
    from pyprojroot.here import here
    wd = here()

    quiz_name = wd / f"ClassFactoryOutput/QuizMaker/L24/l22_24_quiz.xlsx"
    quiz_path = wd / f"ClassFactoryOutput/QuizMaker/"

    sample_size = 5
    quiz_data = load_data(inputDir / quiz_name, sample=sample_size)
    quiz_app(quiz_data, save_results=True, output_dir=quiz_path)
