# %%
"""
ClassFactory Simple Runner
=========================
Run individual ClassFactory modules with minimal configuration.
"""
# %%

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pyprojroot.here import here

from class_factory.ClassFactory import ClassFactory
from class_factory.utils.tools import get_llm

load_dotenv()
wd = here()
user_home = Path.home()

# =============================================================================
# SETUP
# =============================================================================

# Load configuration
with open("class_config.yaml", "r") as file:
    config = yaml.safe_load(file)

class_config = config['PS300']  # Change this to switch courses

# %%
# =============================================================================
# USER CONFIGURATION - Set these variables manually
# =============================================================================

# Lesson configuration
LESSON_NO = 2  # Set your lesson number
LESSON_RANGE = range(1, LESSON_NO + 1)  # Adjust if needed for ConceptWeb/QuizMaker

# Model configuration
MODEL_TYPE = "gemini"  # Options: openai, anthropic, gemini, ollama


# %%
# =============================================================================
# INITIALIZE FACTORY
# =============================================================================
# Run this cell to create the factory instance

llm = get_llm(MODEL_TYPE)
factory = ClassFactory(
    lesson_no=LESSON_NO,
    llm=llm,
    config=class_config,  # Pass the config dict directly
    project_dir=wd,
    lesson_range=LESSON_RANGE,
    verbose=True
)
print(f"Factory initialized for Lesson {LESSON_NO} using {MODEL_TYPE}")

# %%
# =============================================================================
# BEAMERBOT - Run this cell to generate slides
# =============================================================================

specific_guidance = """
None
"""

lesson_objectives = class_config.get('lesson_objectives', {})
beamerbot = factory.create_module("BeamerBot", output_format="latex", verbose=False)
slides = beamerbot.generate_slides(specific_guidance=specific_guidance, lesson_objectives=lesson_objectives)

print("BeamerBot completed!")
print("To save: beamerbot.save_slides(slides)")
print("To publish: beamerbot.publish_slides(user_home / class_config['slideDir'])")

# %%
# =============================================================================
# CONCEPT WEB - Run this cell to build concept map
# =============================================================================

builder = factory.create_module("ConceptWeb", verbose=False, lesson_range=LESSON_RANGE)
builder.build_concept_map(
    directed=False,
    concept_similarity_threshold=0.995,
    dark_mode=True,
    lesson_objectives=None
)
print("ConceptWeb completed!")

# %%
# =============================================================================
# QUIZ MAKER - Run this cell to generate quiz
# =============================================================================

quiz_dir = wd / "data/quizzes/"
quizmaker = factory.create_module("QuizMaker", lesson_range=LESSON_RANGE, prior_quiz_path=quiz_dir, verbose=False)
quiz = quizmaker.make_a_quiz(flag_threshold=0.7, difficulty_level=7)

quizmaker.save_quiz(quiz)
print(f"Quiz saved to {quizmaker.output_dir}")

# %%
