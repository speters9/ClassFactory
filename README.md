Here's an updated README that includes your functionality and explanations based on the example implementations provided:

---

# ClassFactory

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**LLM-powered tools for instructional product generation**

## Overview

ClassFactory is a modular toolkit designed to automate various aspects of lesson and course material generation using language models (LLMs). It offers functionality to create interactive learning resources, including LaTeX Beamer slides, concept maps, and quizzes, all structured around a specified syllabus or lesson plan.

The key modules include:
- **BeamerBot** for automated Beamer slide generation.
- **ConceptWeb** for building concept maps based on lesson readings.
- **QuizMaker** for quiz creation using both lesson content and prior quiz data for comparison.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data               <- Data from various sources (external, raw, processed, etc.).
├── docs               <- Documentation for the project.
├── models             <- Trained and serialized models, model predictions, or model summaries.
├── notebooks          <- Jupyter notebooks. Includes example implementations.
├── pyproject.toml     <- Project configuration file with package metadata and dependencies.
├── reports            <- Generated reports (concept maps, figures, etc.).
└── src                <- Source code for this project.
    ├── beamer_bot     <- Source code for BeamerBot slide generation.
    ├── class_factory  <- Core class for creating modules (BeamerBot, ConceptWeb, QuizMaker).
    ├── concept_web    <- Source code for concept map generation.
    ├── quiz_maker     <- Source code for quiz generation.
    ├── utils          <- Utility functions for OCR, loading documents, etc.
```

---

## How to Use ClassFactory

To get started with ClassFactory, ensure that you have configured your environment correctly (e.g., API keys for LLMs) and set the paths for your project directories.

### Example Implementation (using `run_classfactory.py`)

```python
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from pyprojroot.here import here
from pathlib import Path
from src.class_factory.ClassFactory import ClassFactory

# Load environment variables
OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Define paths
readingDir = Path(os.getenv('readingsDir'))
slideDir = Path(os.getenv('slideDir'))
syllabus_path = Path(os.getenv('syllabus_path'))

# Initialize the language model (LLM)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=OPENAI_KEY,
    organization=OPENAI_ORG,
)

# Create ClassFactory instance
factory = ClassFactory(
    lesson_no=21,
    syllabus_path=syllabus_path,
    reading_dir=readingDir,
    slide_dir=slideDir,
    llm=llm,
    project_dir=here(),
    lesson_range=range(17, 21),
    course_name="American Government"
)

# Generate Beamer slides
beamerbot = factory.create_module("BeamerBot", verbose=False)
slides = beamerbot.generate_slides()
beamerbot.save_slides(slides)

# Build concept map
builder = factory.create_module("ConceptWeb", course_name="American Government", lesson_range=range(19, 21))
builder.build_concept_map()

# Create a quiz
quizmaker = factory.create_module("QuizMaker", lesson_range=range(19, 21))
quiz = quizmaker.make_a_quiz()
quizmaker.save_quiz(quiz)
quizmaker.save_quiz_to_ppt(quiz)
```

### Key Modules

#### **BeamerBot**
This module automatically generates LaTeX Beamer slides based on lesson objectives, readings, and prior lessons. It uses an LLM to fill in the slides according to user-specified prompts or default guidance.

```python
beamerbot = factory.create_module("BeamerBot", verbose=True)
slides = beamerbot.generate_slides(specific_guidance="Focus on campaigns and voter behavior.")
beamerbot.save_slides(slides)
```

#### **ConceptWeb**
ConceptWeb builds a visual concept map from lesson materials and readings. It extracts relationships between concepts using LLM-driven text summarization and relationship extraction.

```python
concept_map = factory.create_module("ConceptWeb", course_name="Political Science")
concept_map.build_concept_map()
```

#### **QuizMaker**
QuizMaker generates quiz questions from the readings and objectives. It ensures diversity in questions by comparing newly generated questions with prior quizzes using embedding similarity checks.

```python
quizmaker = factory.create_module("QuizMaker", lesson_range=range(19, 21))
quiz = quizmaker.make_a_quiz(flag_threshold=0.6)
quizmaker.save_quiz(quiz)
quizmaker.save_quiz_to_ppt(quiz)
```

---

### Environment Setup

Make sure to set the following environment variables in your `.env` file:

```
openai_key=<YOUR_OPENAI_API_KEY>
openai_org=<YOUR_OPENAI_ORG>
readingsDir=<PATH_TO_READING_MATERIALS>
slideDir=<PATH_TO_SLIDES>
syllabus_path=<PATH_TO_SYLLABUS>
```

You can install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

---

### Folder Structure

ClassFactory assumes a specific folder structure for input and output data:

```
ClassFactoryOutput/
    ├── BeamerBot/      <- Generated LaTeX files for slides.
    ├── ConceptWeb/     <- Generated concept maps and reports.
    └── QuizMaker/      <- Generated quizzes and presentations.
data/
    ├── quizzes/        <- Previous quizzes for comparison.
reports/
    ├── ConceptWebOutput/ <- Generated concept map visualizations.
    └── figures/        <- Generated graphs, word clouds, etc.
```

---

### Customization and Extensibility

ClassFactory is designed to be modular. You can create new modules by extending the base class and integrating additional functionalities (e.g., interactive simulations or new quiz formats). Each module supports custom input and output directories, so outputs can be flexibly stored or processed further.

### Logging and Debugging

The modules have built-in logging capabilities, with verbosity controlled during initialization. All logging outputs are stored for debugging or tracing operations, especially useful during large-scale batch processing.

---

### Contributing

We welcome contributions! If you'd like to contribute, please fork this repository and submit a pull request. Make sure to include tests for any new functionality and to adhere to the established code structure.

---

This README serves as the guide for setting up and using the ClassFactory to generate instructional content.