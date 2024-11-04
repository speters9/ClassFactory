
import os
from pathlib import Path

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from pyprojroot.here import here

from class_factory.ClassFactory import ClassFactory
from class_factory.utils.tools import reset_loggers

reset_loggers()
wd = here()
user_home = Path.home()

# llm setup
OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Path definitions
readingDir = user_home / os.getenv('readingsDir')
slideDir = user_home / os.getenv('slideDir')
syllabus_path = user_home / os.getenv('syllabus_path')


lesson_no = 29

# %%

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
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
                       lesson_range=range(27, 30),
                       course_name="American Government and Politics")

# %%


############# Build Beamer Slides ################


# %%

specific_guidance = """
I want to discuss economic policy from the standpoint of the subprime mortgage crisis and the subsequent Great Recession.
This presents a good case study demonstrating the uses of all the economic policy levers that were discussed in the readings.

We can then move on to a discussion of subsidies and the CHIPS Act.
"""

beamerbot = factory.create_module("BeamerBot", verbose=False, slide_dir=slideDir)
slides = beamerbot.generate_slides(specific_guidance=specific_guidance)           # Sometimes specific guidance makes the results more generic
# beamerbot.save_slides(slides)

# %%


############# Build Concept Map ################


# %%

builder = factory.create_module("ConceptWeb",
                                course_name="American Government",
                                verbose=True)

builder.build_concept_map(directed=False)

# %%


############# Build a Quiz ################


# %%

quizDir = wd / "data/quizzes/"
quizmaker = factory.create_module("QuizMaker",
                                  course_name="American Government and Politics",
                                  lesson_range=range(25, 26),
                                  prior_quiz_path=quizDir,
                                  verbose=True)
# results_dir=wd / "ClassFactoryOutput/QuizMaker/quiz_results"
# quizmaker.assess_quiz_results()  # results_dir=results_dir)

# %%
quiz = quizmaker.make_a_quiz(flag_threshold=0.6, difficulty_level=6)
# quizmaker.save_quiz(quiz)

# quizmaker.save_quiz_to_ppt(quiz)


# %%
template_path = wd/"references/quiz_slide_template.pptx"  # if desired; often multiple questions exceed template boundary
quiz_path = wd / f"ClassFactoryOutput/QuizMaker/L{lesson_no}/l17_25_quiz.xlsx"
# quizmaker.save_quiz_to_ppt(excel_file=quiz_path, template_path=template_path)
quizmaker.launch_interactive_quiz(quiz_data=quiz_path, sample_size=10, save_results=True, seed=80920,
                                  output_dir=quiz_path.parent, qr_name="unit_3_4_quiz")
# quizmaker.assess_quiz_results()
