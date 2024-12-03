"""Run classfactory implementation -- all 3 modules below."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from pyprojroot.here import here

from class_factory.ClassFactory import ClassFactory

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


lesson_no = 36

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
                       lesson_range=range(27, 37),
                       verbose=False)

# %%


############# Build Beamer Slides ################


# %%
# Using this markdown format, we can also specify exact verbiage to add on slides
specific_guidance = """
Add each section of the below information to its own slide (the section about what to expect for each lesson should all be on one slide):

---
## Roadmap for the Next Few Lessons:

### Lesson 37: Introduce Exercise (class right before Thanksgiving break)​

    - Provide an overview of the exercise to cadets​
    - Give cadets time to discuss potential approaches and work on their memo​

### Lesson 38: Day 1 of Exercise​

    - Have cadets divide up into groups based on party and/or preferred policy (interest groups can float as necessary).​
    - Then have cadets return to their respective institutions (house, senate, executive branch) to start negotiating with key decision-makers. ​
    - Make sure cadets capture any progress so far, so they are ready to jump in for the next round of the exercise. ​

### Lesson 39: Day 2 of Exercise ​

    - Cadets will work with any members necessary to draft a feasible policy. ​
    - At this point, cadets should know who the key power brokers are and should be focusing on getting the approval of anyone who could potentially sink the bill.​
    - Hold a final vote on the proposed policy​
    - Debrief (if time)​

### Lesson 40: Flex Day​/ Final Review

    - Finish up debrief

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
slides = beamerbot.generate_slides(specific_guidance=specific_guidance)           # Sometimes specific guidance makes the results more generic
# beamerbot.save_slides(slides)

# %%


############# Build Concept Map ################


# %%

builder = factory.create_module("ConceptWeb", verbose=False)

builder.build_concept_map(directed=False)

# %%


############# Build a Quiz ################


# %%

quizDir = wd / "data/quizzes/"
# results_dir = wd / "ClassFactoryOutput/QuizMaker/L35/quiz_results"
quizmaker = factory.create_module("QuizMaker",
                                  lesson_range=range(27, 37),
                                  prior_quiz_path=quizDir,
                                  verbose=False)
# quizmaker.assess_quiz_results()  # results_dir=results_dir)

# %%
quiz = quizmaker.make_a_quiz(flag_threshold=0.6, difficulty_level=8)
quizmaker.save_quiz(quiz)

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
