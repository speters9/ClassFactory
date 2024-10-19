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


lesson_no = 25

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
                       lesson_range=range(22, 26),
                       course_name="American Government and Politics")

# %%


############# Build Beamer Slides ################


# %%

specific_guidance = """
A tension that undergirds this lesson is the extent to which institutions encourage democratic success,
as compared to norms.
"""

beamerbot = factory.create_module("BeamerBot", verbose=False)
slides = beamerbot.generate_slides()  # specific_guidance=specific_guidance)           # Sometimes specific guidance makes the results more generic
beamerbot.save_slides(slides)

# %%


############# Build Concept Map ################


# %%

builder = factory.create_module("ConceptWeb",
                                course_name="American Government",
                                lesson_range=range(22, 26))

builder.build_concept_map()

# %%


############# Build a Quiz ################


# %%

quizDir = wd / "data/quizzes/"
quizmaker = factory.create_module("QuizMaker",
                                  course_name="American Government and Politics",
                                  lesson_range=range(17, 26),
                                  prior_quiz_path=quizDir,
                                  verbose=True)


# %%
quiz = quizmaker.make_a_quiz(flag_threshold=0.6)
# quizmaker.save_quiz(quiz)

# quizmaker.save_quiz_to_ppt(quiz)


# %%
template_path = wd/"references/quiz_slide_template.pptx"  # if desired; often multiple questions exceed template boundary
quiz_path = wd / f"ClassFactoryOutput/QuizMaker/L{lesson_no}/l17_25_quiz.xlsx"
quizmaker.save_quiz_to_ppt(excel_file=quiz_path, template_path=template_path)
quizmaker.launch_interactive_quiz(quiz_data=quiz_path, sample_size=10, save_results=True, seed=80920,
                                  output_dir=quiz_path.parent, qr_name="unit_3_4_quiz")
# quizmaker.assess_quiz_results()
