import os
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from pyprojroot.here import here
from pathlib import Path
from src.class_factory.ClassFactory import ClassFactory

wd = here()

OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Path definitions
readingDir = Path(os.getenv('readingsDir'))
slideDir = Path(os.getenv('slideDir'))
syllabus_path = Path(os.getenv('syllabus_path'))


lesson_no = 21

input_dir = readingDir / f'L{lesson_no}/'
beamer_example = slideDir / f'L{lesson_no-1}.tex'
beamer_output = slideDir / f'L{lesson_no}.tex'

#%%

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
#     temperature=0.2
# )

# Initialize the factory
factory = ClassFactory(lesson_no=lesson_no,
                       syllabus_path=syllabus_path,
                       reading_dir=readingDir,
                       slide_dir=slideDir,
                       llm=llm,
                       project_dir=wd,
                       lesson_range=range(17, 21),
                       course_name="American Government")

#%%
# build slides

specific_guidance = """
The lesson should be structured in a way that discusses big picture ideas about political campaigns.
The slides will, at a minimum, cover the following (using the readings as a reference):
           - How campaigns operate
           - How our party system came to be
           - Individual vote choice
           - The types and determinants of today's polarization​
"""

beamerbot = factory.create_module("BeamerBot", verbose = False)
slides = beamerbot.generate_slides()           # Sometimes specific guidance makes the results more generic
beamerbot.save_slides(slides)

#%%
# build concept map
builder = factory.create_module("ConceptWeb",
                                course_name = "American Government",
                                lesson_range=range(19, 21))

builder.build_concept_map()

#%%
# make a quiz
quizDir = wd / "data/quizzes/"
quizmaker = factory.create_module("QuizMaker",
                                  course_name = "American Government and Politics",
                                  lesson_range=range(19, 21),
                                  prior_quiz_path=quizDir)


quiz = quizmaker.make_a_quiz(flag_threshold=0.6)
quizmaker.save_quiz(quiz)
quizmaker.save_quiz_to_ppt(quiz)
quizmaker.launch_interactive_quiz(quiz_data=quiz, sample_size=7)
