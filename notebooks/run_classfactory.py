"""Run classfactory implementation -- all 3 modules below."""
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


lesson_no = 29

# %%

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.8,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_KEY,
    organization=OPENAI_ORG,
)

# llm = ChatAnthropic(
#     model = "claude-3-5-haiku-latest",
#     temperature=0.5,
#     max_retries=2,
#     api_key=ANTHROPIC_API_KEY
#     )


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
                       course_name="American Government",
                       lesson_range=range(27, 29),
                       verbose=False)

# %%


############# Build Beamer Slides ################


# %%
# Using this markdown format, we can also specify exact verbiage to add on slides
specific_guidance = """
Add each section of the below information to its own slide (the section about what to expect for each lesson should all be on one slide):

---

## A Note on Congressional Committees:

     - Traditionally, the most senior member of the committee from the majority party became the “chair” of a committee.
     - The most senior member of the minority party was called the “ranking member” of the committee.
     - Thousands of bills are introduced in Congress each year; however, only a few hundred are considered by the full House or Senate.
     - After bills are introduced, they are sent to the appropriate committee (and possibly, subcommittee) where the hard work of writing legislation is done. Most bills are never passed out of their committees and must be re-introduced in the next Congress for consideration.
     - Bills “die” in committee for various reasons. Some bills are duplicative; some bills are written to bring attention to issues without expectation of becoming law; some are not practical ideas.
     - Committees use professional staff, and experts representing business, labor, the public and the executive branch to obtain information needed by members in writing legislation.

---
## What's the Big Deal About Immigration?

     - Situation: The Department of Homeland Security has recently reported that over the coming months, a surge of migrants—more than double the average in recent years—will arrive at the southern border seeking asylum.
     - In September 2024, Border Patrol apprehended 101,790 migrants who crossed into the U.S. without authorization.
     - The United States has economic, humanitarian, and security interests in managing flows of asylum seekers.

---

## Policy Exercise Memo:

    - Using the information in the Policy Exercise Folder on Teams and outside research, answer the following prompts in 1-2 pages (single spaced, 12pt font). Submit your answers on Blackboard before class on lesson 38. If you use outside sources, cite them in MLA format. LLM usage follows the syllabus rules.
    - This memo is worth 40 points (40% of the policy exercise grade). 30 points will be graded on content, and 10 points will be graded on grammar and readability.
    - Briefly summarize the issue.
    - What does your position/organization do?
    - What are your objectives for the exercise?
    - What is your initial policy proposal (how are you going to propose to fix the issue)?
    - What will be your strategy to achieve your objectives?
    - Who are your potential allies?
    - What are limitations you have or see facing?
    - What is your best alternative to a negotiated agreement (i.e. what will you do to achieve your objectives if you cannot reach a comprehensive agreement)?

"""

beamerbot = factory.create_module("BeamerBot", verbose=False, slide_dir=slideDir)
slides = beamerbot.generate_slides()           # Sometimes specific guidance makes the results more generic
# beamerbot.save_slides(slides)

# %%


############# Build Concept Map ################


# %%

builder = factory.create_module("ConceptWeb", verbose=False, lesson_range=range(1, 40))

builder.build_concept_map(directed=False, concept_similarity_threshold=0.995)

# %%


############# Build a Quiz ################


# %%

quizDir = wd / "data/quizzes/"
# results_dir = wd / "ClassFactoryOutput/QuizMaker/L35/quiz_results"
quizmaker = factory.create_module("QuizMaker",
                                  lesson_range=range(1, 3),
                                  prior_quiz_path=quizDir,
                                  verbose=True)
# quizmaker.assess_quiz_results()  # results_dir=results_dir)

# %%
quiz = quizmaker.make_a_quiz(flag_threshold=0.6, difficulty_level=9)
# quizmaker.save_quiz(quiz)

# %%

# quizmaker.save_quiz_to_ppt(quiz)

# # or

# template_path = wd/"references/quiz_slide_template.pptx"  # if desired; often multiple questions exceed template boundary
# quizmaker.save_quiz_to_ppt(excel_file=quiz_path, template_path=template_path)


# %%
# quiz_path = wd / f"ClassFactoryOutput/QuizMaker/L36/l27_36_quiz.xlsx"
results_dir = wd / "ClassFactoryOutput/QuizMaker/L35/quiz_results"

quizmaker.launch_interactive_quiz(quiz_data=quiz, sample_size=10,
                                  save_results=True, seed=80920,
                                  qr_name="reading_quiz_3_review")
# quizmaker.assess_quiz_results(results_dir=results_dir)
