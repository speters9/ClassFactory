# %%
"""
ClassFactory Simple Runner
=========================
Run individual ClassFactory modules with minimal configuration.
"""
# %%
import os
import shutil
from pathlib import Path

import yaml
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

# =============================================================================
# USER INPUT
# =============================================================================

print("ClassFactory Simple Runner. You will be prompted for inputs on what to run. Make sure you have a config file with the necessary settings.\n")

print("\nInput the lesson number you wish to process (e.g., 1, 2, 3, ...).")
LESSON_NO = int(input("Enter the lesson number: "))

print("\nSelect LLM type:")
print("Options: openai, anthropic, gemini, ollama")
MODEL_TYPE = input("Enter LLM type (openai/anthropic/gemini/ollama) [default: gemini]: ").strip() or "gemini"

print("\nSelect module:")
print("1. BeamerBot")
print("2. ConceptWeb")
print("3. QuizMaker")
module_choice = input("Enter choice (1-3): ").strip()

# For modules that support ranges, get range input
if module_choice in ["2", "3"]:
    module_name = "ConceptWeb" if module_choice == "2" else "QuizMaker"
    start_lesson = input(f"Enter start lesson for {module_name} [default: 1]: ").strip()
    start_lesson = int(start_lesson) if start_lesson else 1
    end_lesson = input(f"Enter end lesson for {module_name} [default: {LESSON_NO}]: ").strip()
    end_lesson = int(end_lesson) if end_lesson else LESSON_NO
    LESSON_RANGE = range(start_lesson, end_lesson + 1)
    print(f"Using lesson range: {start_lesson} to {end_lesson}")
else:
    LESSON_RANGE = range(1, LESSON_NO + 1)


# =============================================================================
# SETUP
# =============================================================================

# Load configuration
with open("class_config.yaml", "r") as file:
    config = yaml.safe_load(file)

class_config = config['PS460']  # Change this to switch courses

# Extract paths and settings
slide_dir = user_home / class_config['slideDir']
syllabus_path = user_home / class_config['syllabus_path']
readingsDir = user_home / class_config['reading_dir']
is_tabular_syllabus = class_config['is_tabular_syllabus']
config_lesson_objectives = class_config['lesson_objectives']
course_name = class_config['course_title'] or "political science"

# API keys
OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')
ANTHROPIC_API_KEY = os.getenv("anthropic_api_key")
GEMINI_KEY = os.getenv('gemini_api_key')


def get_llm(model_type: str = "gemini"):
    """Select and configure the LLM to use."""
    model_type = model_type.lower()

    if model_type == "openai":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=OPENAI_KEY)
    elif model_type == "anthropic":
        return ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0.4, max_retries=2, api_key=ANTHROPIC_API_KEY)
    elif model_type == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4, max_retries=2, api_key=GEMINI_KEY)
    elif model_type == "ollama":
        return Ollama(model="mistral", temperature=0.3)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def publish_slides(lesson_no: int, source_dir: Path = None, dest_dir: Path = None):
    """Copy finalized slides from ClassFactory output to master slide directory."""
    if source_dir is None:
        source_dir = wd / f"ClassFactoryOutput/BeamerBot/L{lesson_no}"
    if dest_dir is None:
        dest_dir = slide_dir

    source_file = source_dir / f"L{lesson_no}.tex"
    dest_file = dest_dir / f"L{lesson_no}.tex"

    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created destination directory: {dest_dir}")

    shutil.copy2(source_file, dest_file)
    print(f"Published L{lesson_no}.tex to {dest_file}")


# Initialize factory
llm = get_llm(MODEL_TYPE)
factory = ClassFactory(
    lesson_no=LESSON_NO,
    syllabus_path=syllabus_path,
    reading_dir=readingsDir,
    llm=llm,
    project_dir=wd,
    course_name=course_name,
    lesson_range=LESSON_RANGE,
    tabular_syllabus=is_tabular_syllabus,
    verbose=True
)

# =============================================================================
# RUN SELECTED MODULE
# =============================================================================

if module_choice == "1":
    # BeamerBot
    print("\nRunning BeamerBot...")

    specific_guidance = """
    - Just before the "Where We Are in the Course" slide, insert a slide titled "Current Event". The current event slide can be blank.
    - **DO NOT USE lesson objectives that are contained in any of the readings**
    - Remember, this is a Beamer presentation, so all text and fonts should be in LaTeX format.
    - **For this lesson only** you are authorized to create your own lesson objectives, if none are provided. Still, all lesson content should come from the assigned readings.
    - Don't just describe the readings; synthesize them into broader themes.
    """

    lesson_objectives = {} or config_lesson_objectives
    beamerbot = factory.create_module("BeamerBot", verbose=False, slide_dir=slide_dir)
    slides = beamerbot.generate_slides(specific_guidance=specific_guidance, lesson_objectives=lesson_objectives)

    print("BeamerBot completed!")
    print("To save: beamerbot.save_slides(slides). \nThis will save locally in ClassFactoryOutput/BeamerBot/ for inspection.")
    print("To publish: publish_slides(LESSON_NO). \nThis will push to your master slide directory.")

elif module_choice == "2":
    # ConceptWeb
    print("\nRunning ConceptWeb...")

    builder = factory.create_module("ConceptWeb", verbose=False, lesson_range=LESSON_RANGE)
    builder.build_concept_map(directed=False, concept_similarity_threshold=0.995, dark_mode=True, lesson_objectives=None)


elif module_choice == "3":
    # QuizMaker
    print("\nRunning QuizMaker...")

    quiz_dir = wd / "data/quizzes/"
    quizmaker = factory.create_module("QuizMaker", lesson_range=LESSON_RANGE, prior_quiz_path=quiz_dir, verbose=False)
    quiz = quizmaker.make_a_quiz(flag_threshold=0.7, difficulty_level=7)

    quizmaker.save_quiz(quiz)
    print(f"Quiz saved to  {quizmaker.output_dir}")

else:
    print("Invalid choice.")

print(f"\nCompleted for Lesson {LESSON_NO}!")

# %%
