import logging
import os
from pathlib import Path

from langchain.globals import set_verbose
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from pyprojroot.here import here

from src.class_factory.ClassFactory import ClassFactory

logging.basicConfig(level=logging.WARNING)

wd = here()
# llm setup
OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')
set_verbose(False)
# Path definitions
readingDir = Path(os.getenv('readingsDir'))
slideDir = Path(os.getenv('slideDir'))
syllabus_path = Path(os.getenv('syllabus_path'))


lesson_no = 22

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
                       lesson_range=range(17, 23),
                       course_name="American Government")

# %%
# build slides

specific_guidance = """
The lesson should address a tension that exists between our current lesson and the intent of the Founders we discussed in Unit 1.
We discussed how the government was intentionally designed to move slowly, in order to preserve individual liberty.
What about when the government's slow movement hinders the full realization of liberty in the country?
"""

beamerbot = factory.create_module("BeamerBot", verbose=False)
slides = beamerbot.generate_slides(specific_guidance=specific_guidance)           # Sometimes specific guidance makes the results more generic
# beamerbot.save_slides(slides)

# %%
# build concept map
builder = factory.create_module("ConceptWeb",
                                course_name="American Government",
                                lesson_range=range(17, 22))

builder.build_concept_map()

# %%
# make a quiz
quizDir = wd / "data/quizzes/"
quizmaker = factory.create_module("QuizMaker",
                                  course_name="American Government and Politics",
                                  lesson_range=range(17, 22),
                                  prior_quiz_path=quizDir)


quiz = quizmaker.make_a_quiz(flag_threshold=0.6)
quizmaker.save_quiz(quiz)
# quizmaker.save_quiz_to_ppt(quiz)

# %%

quiz_path = wd / f"ClassFactoryOutput/QuizMaker/L{lesson_no}/l17_21_quiz.xlsx"
quizmaker.launch_interactive_quiz(quiz_data=quiz_path, sample_size=7)
