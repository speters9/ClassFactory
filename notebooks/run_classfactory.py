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


LESSON_NO = 16

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

# Initialize the factory
factory = ClassFactory(lesson_no=LESSON_NO,
                       syllabus_path=syllabus_path,
                       reading_dir=readingDir,
                       llm=llm,
                       project_dir=wd,
                       course_name="Foreign Policy",
                       lesson_range=range(1, LESSON_NO+1),
                       verbose=False)

# %%


############# Build Beamer Slides ################


# %%
# Using this markdown format, we can also specify exact verbiage to add on slides
specific_guidance = """
- After the "Lesson Objectives" slide, add a slide titled "Stand and Deliver". The Stand and Deliver slide can be blank.
- **DO NOT USE lesson objectives that are contained in any of the readings**
"""

lesson_objectives = {
    "16": """
            Describe the role played by the Intelligence agencies in the foreign policy making process.
            Explain the challenges of interagency coordination within the intelligence community.
            Articulate the tensions between politics and intelligence.
            """
}

beamerbot = factory.create_module(
    "BeamerBot", verbose=False, slide_dir=slideDir)

slides = beamerbot.generate_slides(specific_guidance=specific_guidance,
                                   lesson_objectives=lesson_objectives,
                                   tabular_syllabus=True)
print(slides)
# beamerbot.save_slides(slides, output_dir=slideDir)


# %%


############# Build Concept Map ################


# %%
lesson_objectives = {
    "15": """
        Articulate the roles of the Defense Department in the foreign policy making process.
        Explain the direct and indirect ways in which DoD can influence policy.
        Understand the challenges of interagency coordination in foreign policymaking.
        """,
    "14": """
        Articulate the roles of the State Department in the foreign policy making process.
        Understand the challenges of interagency coordination in foreign policymaking.
        """,
    "12": """
        Summarize the unique roles, authorities, and powers of the US president in the foreign policy making process.
        Articulate the limits of presidential authorities in making foreign policy.
        """,
    "11": """
        Define and explain the Organizational Process Model.
        Contrast the Organizational Process Model with the other models we've covered.
        """,
    "10": """
        Define and explain the rational actor model in foreign policy decision-making.
        Compare the rational actor model to other decision-making models.
    """,
    "9": """
        Define and explain the influence of small groups and elites in foreign policy.
        Analyze the dynamics of group decision-making in the context of international relations.
    """,
    "8": """
        Define and explain the bureaucratic politics model.
        Compare the bureaucratic politics model to other decision-making models.
    """,

}

builder = factory.create_module("ConceptWeb",
                                verbose=False,
                                lesson_range=range(8, 12))
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
                                  lesson_range=range(8, 12),
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
