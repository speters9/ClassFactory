"""Run classfactory implementation -- all 3 modules below."""
# %%
import os
from pathlib import Path

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
readingDir = user_home / os.getenv('readingsDir')
slideDir = user_home / os.getenv('slideDir')
syllabus_path = user_home / os.getenv('syllabus_path')


lesson_no = 6

# %%

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0.4,
#     api_key=OPENAI_KEY,
# )

llm = ChatAnthropic(
    model="claude-3-5-haiku-latest",
    temperature=0.4,
    max_retries=2,
    api_key=ANTHROPIC_API_KEY
)

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash-8b",
#     temperature=0.4,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     api_key=GEMINI_KEY
# )

# llm = Ollama(
#     model="mistral",
#     #model="llama3.1",
#     temperature=0.3
# )

# Initialize the factory
factory = ClassFactory(lesson_no=lesson_no,
                       syllabus_path=syllabus_path,
                       reading_dir=readingDir,
                       llm=llm,
                       project_dir=wd,
                       course_name="Foreign Policy",
                       lesson_range=range(1, 7),
                       verbose=False)

# %%


############# Build Beamer Slides ################


# %%
# Using this markdown format, we can also specify exact verbiage to add on slides
specific_guidance = """
- Before the "Discussion Question" slide, add a slide titled "Stand and Deliver". The Stand and Deliver slide can be blank.
- **DO NOT USE the lesson objectives if the lesson objectives contain readings**
"""

lesson_objectives = {
    "6": """State the legal requirements and purposes of the National Security Strategy.​
    Analyze recent National Security Strategies and compare and contrast their structure, content, messages, strengths, and critiques.​
    Evaluate the key continuities and differences between the 2022 National Security Strategy and previous administrations' strategies. """
}

beamerbot = factory.create_module(
    "BeamerBot", verbose=False, slide_dir=slideDir)
slides = beamerbot.generate_slides(specific_guidance=specific_guidance,
                                   lesson_objectives=lesson_objectives)           # Sometimes specific guidance makes the results more generic
print(slides)
# beamerbot.save_slides(slides, output_dir=slideDir)


# %%


############# Build Concept Map ################


# %%

builder = factory.create_module("ConceptWeb", verbose=False, lesson_range=range(1, 11))

builder.build_concept_map(directed=False, concept_similarity_threshold=0.995, dark_mode=False)

# %%


############# Build a Quiz ################


# %%

quizDir = wd / "data/quizzes/"
# results_dir = wd / "ClassFactoryOutput/QuizMaker/L35/quiz_results"
quizmaker = factory.create_module("QuizMaker",
                                  lesson_range=range(1, 7),
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
quiz_path = wd / f"ClassFactoryOutput/QuizMaker/L6/l1_6_quiz.xlsx"
# results_dir = wd / "ClassFactoryOutput/QuizMaker/L6/quiz_results"

quizmaker.launch_interactive_quiz(quiz_data=quiz,
                                  sample_size=10,
                                  save_results=True,
                                  seed=42,
                                  qr_name="quiz_review")
# quizmaker.assess_quiz_results()  # If analyzing a different quiz, simply provide the directory containing saved quizzes as `results_dir`

# %%
