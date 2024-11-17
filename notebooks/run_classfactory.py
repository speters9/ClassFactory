
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from pyprojroot.here import here

from class_factory.ClassFactory import ClassFactory
from class_factory.utils.tools import reset_loggers

load_dotenv()
wd = here()
user_home = Path.home()

# llm setup
OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Path definitions
readingDir = user_home / os.getenv('readingsDir')
slideDir = user_home / os.getenv('slideDir')
syllabus_path = user_home / os.getenv('syllabus_path')


lesson_no = 34

# %%

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_KEY,
    organization=OPENAI_ORG,
)
# llm = Ollama(
#     model="llama3.1",
#     #model="gemma2",
#     temperature=0.2
# )

# Initialize the factory
factory = ClassFactory(lesson_no=lesson_no,
                       syllabus_path=syllabus_path,
                       reading_dir=readingDir,
                       llm=llm,
                       project_dir=wd,
                       course_name="American Government",
                       lesson_range=range(30, 33),
                       verbose=False)

# %%


############# Build Beamer Slides ################


# %%

specific_guidance = """
We didn't fully cover the chips act last time, so I want to begin the lesson with the trade policy slides from last lesson
before going into national security policy this lesson.
"""

beamerbot = factory.create_module("BeamerBot", verbose=False, slide_dir=slideDir)
slides = beamerbot.generate_slides()           # Sometimes specific guidance makes the results more generic
# beamerbot.save_slides(slides)

# %%


############# Build Concept Map ################


# %%

builder = factory.create_module("ConceptWeb", verbose=True)

builder.build_concept_map(directed=False)

# %%


############# Build a Quiz ################


# %%

quizDir = wd / "data/quizzes/"
quizmaker = factory.create_module("QuizMaker",
                                  lesson_range=range(30, 33),
                                  prior_quiz_path=quizDir,
                                  verbose=False)
# results_dir=wd / "ClassFactoryOutput/QuizMaker/quiz_results"
# quizmaker.assess_quiz_results()  # results_dir=results_dir)

# %%
quiz = quizmaker.make_a_quiz(flag_threshold=0.6, difficulty_level=8)
# quizmaker.save_quiz(quiz)

# %%

# quizmaker.save_quiz_to_ppt(quiz)

# # or

# template_path = wd/"references/quiz_slide_template.pptx"  # if desired; often multiple questions exceed template boundary
# quizmaker.save_quiz_to_ppt(excel_file=quiz_path, template_path=template_path)


# %%
# quiz_path = wd / f"ClassFactoryOutput/QuizMaker/L25/l17_25_quiz.xlsx"
quizmaker.launch_interactive_quiz(quiz_data=quiz, sample_size=5,
                                  save_results=True, seed=80920,
                                  qr_name="natsec_quiz")
# quizmaker.assess_quiz_results()
