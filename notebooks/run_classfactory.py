"""Run classfactory implementation -- all 3 modules below."""
# %%
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pyprojroot.here import here

from class_factory.ClassFactory import ClassFactory

load_dotenv()
wd = here()
user_home = Path.home()

# llm setup
OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')
ANTHROPIC_API_KEY = os.getenv("anthropic_api_key")
GEMINI_KEY = os.getenv('gemini_api_key')


# Path definitions
with open("class_config.yaml", "r") as file:
    config = yaml.safe_load(file)

class_config = config['PS460']
slide_dir = user_home / class_config['slideDir']
syllabus_path = user_home / class_config['syllabus_path']
readingsDir = user_home / class_config['reading_dir']
is_tabular_syllabus = class_config['is_tabular_syllabus']


# %%

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0.4,
#     api_key=OPENAI_KEY,
# )

# llm = ChatAnthropic(
#     model="claude-3-5-haiku-latest",
#     temperature=0.4,
#     max_retries=2,
#     api_key=ANTHROPIC_API_KEY
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.4,
    max_retries=2,
    api_key=GEMINI_KEY
)

# llm = Ollama(
#     model="mistral",
#     #model="llama3.1",
#     temperature=0.3
# )


LESSON_NO = int(input("Enter the lesson number: "))

# Initialize the factory
factory = ClassFactory(lesson_no=LESSON_NO,
                       syllabus_path=syllabus_path,
                       reading_dir=readingsDir,
                       llm=llm,
                       project_dir=wd,
                       course_name="civil-military relations",
                       lesson_range=range(1, LESSON_NO+1),
                       tabular_syllabus=is_tabular_syllabus,
                       verbose=False)

# %%


############# Build Beamer Slides ################


# %%


# Using this markdown format, we can also specify exact verbiage to add on slides
specific_guidance = """
- Just before the "Lesson Objectives" slide, insert a slide titled "Current Event". The Current Event slide can be blank.
- **DO NOT USE lesson objectives that are contained in any of the readings**
"""

lesson_objectives = {
    "2": """
        Explain the problem of civilian control.
        Understand why the problem of civilian control is so difficult.
    """,
}

beamerbot = factory.create_module(
    "BeamerBot", verbose=False, slide_dir=slide_dir)

slides = beamerbot.generate_slides(specific_guidance=specific_guidance,
                                   lesson_objectives=lesson_objectives)
print(slides)
beamerbot.save_slides(slides, output_dir=slide_dir)


# %%


############# Build Concept Map ################


# %%
lesson_objectives = {

    "2": """
        Explain the problem of civilian control.
        Understand why the problem of civilian control is so difficult.
    """,
}

builder = factory.create_module("ConceptWeb",
                                verbose=False,
                                lesson_range=range(0, 3))
# %%

builder.build_concept_map(
    directed=False,
    concept_similarity_threshold=0.995,
    dark_mode=True,
    lesson_objectives=lesson_objectives)

# %%


############# Build a Quiz ################


# %%

quizDir = wd / "data/quizzes/"
# results_dir = wd / "ClassFactoryOutput/QuizMaker/L35/quiz_results"
quizmaker = factory.create_module("QuizMaker",
                                  lesson_range=range(0, 3),
                                  prior_quiz_path=quizDir,
                                  verbose=False)

# %%
quiz = quizmaker.make_a_quiz(flag_threshold=0.7, difficulty_level=9)
print(quiz)
# quizmaker.save_quiz(quiz)

# %%


# quizmaker.launch_interactive_quiz(quiz_data=quiz)
# quizmaker.assess_quiz_results()  # results_dir=results_dir)
# %%

# quizmaker.save_quiz_to_ppt(quiz)

# # or

# template_path = wd/"references/quiz_slide_template.pptx"  # if desired; often multiple questions exceed template boundary
# quizmaker.save_quiz_to_ppt(excel_file=quiz_path, template_path=template_path)


# %%
quiz_path = wd / f"ClassFactoryOutput/QuizMaker/L8/l1_7_quiz.xlsx"
# results_dir = wd / "ClassFactoryOutput/QuizMaker/L6/quiz_results"

quizmaker.launch_interactive_quiz(quiz_data=quiz_path,
                                  sample_size=10,
                                  save_results=True,
                                  seed=8675309,
                                  qr_name="unit1_review")
# quizmaker.assess_quiz_results()  # If analyzing a different quiz, simply provide the directory containing saved quizzes as `results_dir`

# %%
