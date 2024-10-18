import logging
import os
from pathlib import Path

from langchain.globals import set_verbose
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from pyprojroot.here import here

from src.class_factory.ClassFactory import ClassFactory
from src.utils.tools import reset_loggers

reset_loggers()
wd = here()
# llm setup
OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Path definitions
readingDir = Path(os.getenv('readingsDir'))
slideDir = Path(os.getenv('slideDir'))
syllabus_path = Path(os.getenv('syllabus_path'))


lesson_no = 24

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
#     temperature=0.2
# )

# Initialize the factory
factory = ClassFactory(lesson_no=lesson_no,
                       syllabus_path=syllabus_path,
                       reading_dir=readingDir,
                       slide_dir=slideDir,
                       llm=llm,
                       project_dir=wd,
                       lesson_range=range(22, 25),
                       course_name="American Government and Politics")

# %%

############# Build Beamer Slides ################


# %%

specific_guidance = """
In addition to covering the important concepts from the readings, the lesson should address a tension that exists
between free expression of personal liberties and the possibility that such free expressions
may interfere with the liberties of others. To limit free exercise of our liberties undermines the very protections of liberty, even while
preserving the liberties of others. How do we reconcile this tension?
"""

beamerbot = factory.create_module("BeamerBot", verbose=False)
slides = beamerbot.generate_slides()  # specific_guidance=specific_guidance)           # Sometimes specific guidance makes the results more generic
beamerbot.save_slides(slides)

# %%

############# Build Concept Map ################


# %%

builder = factory.create_module("ConceptWeb",
                                course_name="American Government",
                                lesson_range=range(22, 24))

builder.build_concept_map()

# %%

############# Build a Quiz ################


# %%

quizDir = wd / "data/quizzes/"
quizmaker = factory.create_module("QuizMaker",
                                  course_name="American Government and Politics",
                                  lesson_range=range(22, 25),
                                  prior_quiz_path=quizDir,
                                  verbose=True)


# %%
quiz = quizmaker.make_a_quiz(flag_threshold=0.6)
quizmaker.save_quiz(quiz)

# %%
# quiz_excel_path = wd / f"ClassFactoryOutput/QuizMaker/L22/l17_21_quiz.xlsx"

quiz_path = wd / f"ClassFactoryOutput/QuizMaker/L{lesson_no}/l22_24_quiz.xlsx"
quizmaker.save_quiz_to_ppt(excel_file=quiz_path)

# %%

quiz_path = wd / f"ClassFactoryOutput/QuizMaker/L{lesson_no}/l22_24_quiz.xlsx"
quizmaker.launch_interactive_quiz(quiz_data=quiz_path, sample_size=7, seed=8675309)
