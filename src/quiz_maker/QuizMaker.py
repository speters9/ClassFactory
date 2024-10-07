"""
This script automates the generation of quiz questions for an undergraduate-level political science course.
The quiz is based on lesson objectives, lesson readings, and an existing midterm review, and it checks for
similarity to previous questions. The questions are generated using a language model and returned in multiple formats,
including multiple-choice, true/false, and fill-in-the-blank types.

Workflow:
1. **Data Loading**: Loads lesson readings, objectives, and the existing midterm questions.
2. **Quiz Generation**: Uses a pre-defined prompt to generate a variety of question types (multiple choice, true/false,
   fill-in-the-blank) based on the readings and objectives.
3. **Similarity Check**: Compares the generated questions to the existing midterm questions using sentence embeddings
   to ensure minimal overlap.
4. **Question Flagging**: Flags questions that are too similar to the midterm questions based on a similarity threshold.
5. **Saving the Output**: Outputs the final set of quiz questions in Excel format, excluding flagged questions.

Dependencies:
- This script requires access to an OpenAI API key for generating questions via a language model.
- Ensure the necessary environment variables (`openai_key`, `openai_org`, `syllabus_path`, `readingsDir`, etc.) are set.
- Torch and SentenceTransformer are used for similarity checking, while pandas is used for saving the output.

The expected input files (e.g., lesson readings and syllabus) are organized in directories specified by the environment
variables. The output is saved in an Excel file ready for review and further editing.
"""


# base libraries
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union
import pandas as pd
import logging
# embedding check for similarity against true questions
import torch
# env setup
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
# llm chain setup
from sentence_transformers import SentenceTransformer, util
# ppt setup
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
# self-defined utils
from src.utils.load_documents import load_lessons, extract_lesson_objectives
from src.utils.response_parsers import Quiz
from src.utils.tools import logger_setup

from src.quiz_maker.quiz_prompts import quiz_prompt
from src.quiz_maker.quiz_to_app import quiz_app
from pyprojroot.here import here

load_dotenv()

OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Path definitions
wd = here()
syllabus_path = Path(os.getenv('syllabus_path'))
readingDir = Path(os.getenv("readingsDir"))

outputDir = wd / "data/quizzes/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%


# QuizMaker class definition
class QuizMaker:
    def __init__(self, llm, syllabus_path: Path, reading_dir: Path, output_dir: Path,
                 prior_quiz_path: Path, lesson_range: range, quiz_prompt: str = quiz_prompt, device=None,
                 course_name: str = 'Political Science', verbose=True):
        """
        Initialize QuizMaker with the necessary paths, LLM, and other configurations.

        Args:
            llm: The language model instance for generating quiz questions.
            syllabus_path (Path): Path to the syllabus file.
            reading_dir (Path): Directory where lesson readings are stored.
            output_dir (Path): Directory where the generated quiz will be saved.
            prior_quiz_path (Path): Path to the previous quiz for similarity checking.
            device: The device for sentence embeddings (CPU or GPU).
        """
        self.llm = llm
        self.syllabus_path = syllabus_path
        self.reading_dir = reading_dir
        self.output_dir = output_dir
        self.prior_quiz_path = prior_quiz_path
        self.lesson_range = lesson_range
        self.course_name = course_name

        # setup logging
        log_level = logging.INFO if verbose else logging.WARNING
        self.logger = logger_setup(log_level=log_level)

        # Define the LLM prompt template
        self.quiz_parser = JsonOutputParser(pydantic_object=Quiz)
        self.quiz_prompt = quiz_prompt

        # Set device for similarity checking (default to GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # Load prior quiz questions for similarity checking
        self.prior_quiz_questions, self.prior_quizzes = self.load_and_merge_prior_quizzes()

        # Initialize sentence transformer model for similarity checking
        self.model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.rejected_questions = []

    def make_a_quiz(self, flag_threshold: float = 0.7) -> List[Dict]:
        """
        Generate quiz questions based on lesson readings and objectives. Generate questions one lesson at a time

        Args:
            flag_threshold (float): The similarity metric beyond which a proposed quiz question will be rejected. \
                Compares a generated question to a provided set of questions which students have already seen.
        """
        all_readings = []
        objectives = []
        responselist = []

        for lesson_no in self.lesson_range:
            input_dir = self.reading_dir / f'L{lesson_no}/'

            # load_documents to process all readings for the current lesson
            readings = load_lessons(input_dir, lesson_no, recursive=False)
            all_readings.extend(readings)

            # Extract lesson objectives
            objectives_text = extract_lesson_objectives(self.syllabus_path, lesson_no, only_current=True)
            objectives.append(objectives_text)

            # add rejected questions, if any, to our list of things to avoid
            rejected_questions = [q['question'] for q in self.rejected_questions] if self.rejected_questions else []
            questions_not_to_use = list(set(self.prior_quiz_questions + rejected_questions))

            chain = self.build_quiz_chain()
            # Generate questions using the LLM
            response = chain.invoke({
                "course_name": self.course_name,
                "objectives": objectives_text,
                "information": readings,
                'prior_quiz_questions': questions_not_to_use
            })

            # if string, remove the code block indicators (```json and ``` at the end)
            if isinstance(response, str):
                response_cleaned = response.replace('```json\n', '').replace('\n```', '')
                quiz_questions = json.loads(response_cleaned)
            else:
                quiz_questions = response

            if isinstance(quiz_questions, dict):
                # Flatten the questions and add a 'type' key
                for question_type, questions in quiz_questions.items():
                    for question in questions:
                        # Add the question type as a key to each question
                        question['type'] = question_type
                        # Add the updated question to responselist
                        responselist.append(question)
            else:
                responselist.extend(quiz_questions)

        # Extract just the question text for similarity checking
        generated_question_texts = [q['question'] for q in responselist]

        # Check for similarities
        flagged_questions = self.check_question_similarity(generated_question_texts, threshold=flag_threshold)

        # Separate flagged and scrubbed questions
        scrubbed_questions, flagged_list = self.separate_flagged_questions(responselist, flagged_questions)
        self.rejected_questions.extend(flagged_list)

        return scrubbed_questions

    def build_quiz_chain(self):
        # Construct the prompt template
        combined_template = PromptTemplate.from_template(self.quiz_prompt)
        # Create the chain with the prompt, LLM, and parser
        chain = combined_template | self.llm | self.quiz_parser

        return chain

    def load_and_merge_prior_quizzes(self) -> list:
        """
        Load and merge all prior quiz questions from the Excel files in the prior_quiz_dir.
        Returns a list of quiz questions for similarity checking.
        """
        all_quiz_data = []
        quiz_df = pd.DataFrame()

        # Get all Excel files in the prior_quiz_dir
        for file in self.prior_quiz_path.glob('*.xlsx'):
            df = pd.read_excel(file)
            quiz_df = pd.concat([quiz_df,df], axis=0, ignore_index=True)
            if 'question' in df.columns:
                all_quiz_data.append(df['question'].tolist())

        # Flatten the list of lists into a single list
        merged_questions = [item for sublist in all_quiz_data for item in sublist]
        quiz_df = quiz_df.reset_index(drop=True)

        return merged_questions, quiz_df

    # Function to check similarity between generated questions and main quiz questions
    def check_question_similarity(self, generated_questions, threshold=0.6) -> List[Dict]:
        """
        Check similarity between generated questions and prior quiz questions.
        Returns a list of flagged questions with similarity scores and their indices.
        """
        flagged_questions = []

        # Get embeddings for the newly generated questions and prior quiz questions
        generated_embeddings = self.model.encode(generated_questions, convert_to_tensor=True, device=self.device)
        prior_quiz_embeddings = self.model.encode(self.prior_quiz_questions, convert_to_tensor=True, device=self.device)

        # Compare each generated question to prior quiz questions
        for i, gen_question in enumerate(generated_questions):
            cosine_scores = util.pytorch_cos_sim(generated_embeddings[i], prior_quiz_embeddings)
            max_score = float(cosine_scores.max())

            if max_score > threshold:
                flagged_questions.append({'index': i, 'question': gen_question, 'similarity': max_score})

        return flagged_questions

    def separate_flagged_questions(self, questions, flagged_questions) -> Tuple[List, List]:
        """
        Separate the flagged questions from the valid (scrubbed) questions based on similarity check.

        Args:
            questions (list): List of all generated quiz questions.
            flagged_questions (list): List of flagged question objects with index and similarity.

        Returns:
            Tuple: Two lists - (scrubbed_questions, flagged_questions).
        """
        scrubbed_questions = []
        flagged_list = []

        flagged_indices = {f['index'] for f in flagged_questions}  # Get set of flagged indices

        for i, question in enumerate(questions):
            if i in flagged_indices:
                flagged_list.append(question)
            else:
                scrubbed_questions.append(question)

        return scrubbed_questions, flagged_list

    def save_quiz(self, quiz: List[Dict]) -> None:
        """Save to excel"""
        final_quiz = pd.DataFrame(quiz)

        col_order = ['type', 'question', 'A)', 'B)', 'C)', 'D)', 'correct_answer']
        final_quiz = final_quiz[col_order].reset_index(drop=True)
        final_quiz.to_excel(self.output_dir / f"l{min(self.lesson_range)}_{max(self.lesson_range)}_quiz.xlsx", index=False)

    def save_quiz_to_ppt(self, quiz: List[Dict] = None, excel_file: Path = None) -> None:
        """
        Save quiz questions to a PowerPoint presentation.

        Args:
            quiz (List[Dict], optional): The list of quiz dictionaries. Defaults to None.
            excel_file (Path, optional): The path to an Excel file containing quiz questions. Defaults to None.
        """
        # Load from Excel if an Excel file path is provided
        if excel_file:
            df = pd.read_excel(excel_file)
        elif quiz:
            # Convert the list of dicts into a DataFrame
            df = pd.DataFrame(quiz)
        else:
            raise ValueError("Either 'quiz' or 'excel_file' must be provided.")

        # Create a new PowerPoint presentation
        prs = Presentation()

        # Function to add a slide with a title and content
        def add_slide(prs, title, content, answer=None, is_answer=False, bg_color=(255, 255, 255)):
            slide_layout = prs.slide_layouts[1]  # Layout with title and content
            slide = prs.slides.add_slide(slide_layout)

            # Set slide background color
            background = slide.background
            fill = background.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(bg_color[0], bg_color[1], bg_color[2])

            # Set title text and font size
            title_placeholder = slide.shapes.title
            title_placeholder.text = title
            title_text_frame = title_placeholder.text_frame
            title_text_frame.paragraphs[0].font.size = Pt(24)  # Set title font size

            # Set content text and font size
            text_placeholder = slide.shapes.placeholders[1]
            text_placeholder.text = content
            content_text_frame = text_placeholder.text_frame
            content_text_frame.paragraphs[0].font.size = Pt(32)  # Set content font size

            if is_answer and answer:
                # For answers, add the answer text at the end and adjust the font size
                text_placeholder.text += f"\n\nAnswer: {answer}"
                content_text_frame.paragraphs[0].font.size = Pt(32)  # Adjust answer font size

        # Loop through each question and add to the presentation
        for i, row in df.iterrows():
            # Get question text
            question_text = row['question']

            # Prepare choices (exclude C and D if they are blank)
            choices = f"A) {row['A)']}\nB) {row['B)']}"
            if pd.notna(row['C)']) and row['C)'].strip():  # Only add C if it is not blank
                choices += f"\nC) {row['C)']}"
            if pd.notna(row['D)']) and row['D)'].strip():  # Only add D if it is not blank
                choices += f"\nD) {row['D)']}"

            # Add a slide for the question
            add_slide(prs, f"Question {i + 1}", question_text + "\n\n" + choices, bg_color=(255, 255, 255))

            # Add a slide for the answer
            correct_answer = row['correct_answer']
            if correct_answer in ['A', 'B', 'C', 'D']:
                answer_text = row[f'{correct_answer})']
                add_slide(prs, f"Answer to Question {i + 1}", question_text,
                          answer=f"{correct_answer}: {answer_text}", is_answer=True, bg_color=(255, 255, 255))
            else:
                add_slide(prs, f"Answer to Question {i + 1}", question_text,
                          answer=f"{correct_answer}", is_answer=True, bg_color=(255, 255, 255))

        # Save the PowerPoint presentation
        ppt_path = self.output_dir / f"quiz_presentation_{min(self.lesson_range)}_{max(self.lesson_range)}.pptx"
        prs.save(ppt_path)
        print(f"Presentation saved at {ppt_path}")

    def launch_interactive_quiz(self, quiz_data: Union[pd.DataFrame, Path, str, List[Dict]] = None, sample_size: int = 5) -> None:
        """
        Launch the interactive quiz using Gradio, using either preloaded quiz data or
        dynamically generated quiz data from the class.

        Args:
            quiz_data (Union[pd.DataFrame, Path, str, List[Dict]], optional): The quiz data to be used for the interactive quiz.
                                                                             If not provided, it will use generated quiz data.
            sample_size (int, Default=5): The number of questions to sample from the large df of questions.
        """

        # If quiz_data is a List of Dicts, convert it to a DataFrame
        if isinstance(quiz_data, list) and isinstance(quiz_data[0], dict):  # Check if it's a List[Dict]
            quiz = pd.DataFrame(quiz_data)

        # If quiz_data is a path or string, read the file
        elif isinstance(quiz_data, (Path, str)):
            quiz = pd.read_excel(quiz_data)

        # If quiz_data is None, generate the quiz dynamically
        elif quiz_data is None:
            self.logger.warning("No quiz identified. Generating new quiz from lesson documents")
            quiz_generated = self.make_a_quiz()
            quiz = pd.DataFrame(quiz_generated)

        # If quiz_data is already a DataFrame, use it directly
        elif isinstance(quiz_data, pd.DataFrame):
            quiz = quiz_data

        else:
            raise ValueError("Invalid type for quiz_data. It must be either a DataFrame, path to an Excel file, or a list of dictionaries.")

        quiz_sampled = quiz.sample(sample_size)
        quiz_app(quiz_sampled)


# %%
if __name__ == "__main__":

    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0.1,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     api_key=OPENAI_KEY,
    #     organization=OPENAI_ORG,
    # )
    llm = Ollama(
        model="llama3.1",
        temperature=0.3
    )
    lesson_no = 20

    # Path definitions
    readingDir = Path(os.getenv('readingsDir'))
    slideDir = Path(os.getenv('slideDir'))
    syllabus_path = Path(os.getenv('syllabus_path'))

    maker = QuizMaker(llm=llm,
                      syllabus_path=syllabus_path,
                      reading_dir=readingDir,
                      output_dir=wd/"ClassFactoryOutput/QuizMaker",
                      quiz_prompt=quiz_prompt,
                      lesson_range=range(19, 21),
                      course_name = "American Government",
                      prior_quiz_path=outputDir)

    quiz = maker.make_a_quiz()
    # maker.save_quiz_to_ppt(quiz)

    # maker.save_quiz(quiz)
    #maker.launch_interactive_quiz(wd/"data/processed/l19_quiz.xlsx")
