# ClassFactory

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**AI-powered tools for instructional product generation**

## Overview

ClassFactory is a modular toolkit designed to automate various aspects of lesson and course material generation leveraging the generative capaciites of large language models (LLMs). It offers functionality to create interactive learning resources, including LaTeX Beamer slides, concept maps, and quizzes, all structured around a specified syllabus or lesson plan.

The key modules include:
- **BeamerBot** for automated LaTeX Beamer slide generation.
- **ConceptWeb** for building concept maps (a form of knowledge graph) based on lesson readings. This is best used as a tool to compare how concepts from prior lessons relate to later lessons.
- **QuizMaker** for quiz creation using both lesson content and prior quiz data for comparison. The path to a prior quiz can be passed in during module creation, to avoid question duplication or leakage.

## Documentation
Full project documentation is located [here](https://speters9.github.io/ClassFactory/)

## Project Organization

```
├── LICENSE             <- Open-source license if one is chosen
├── README.md           <- The top-level README for developers using this project.
├── docs                <- Documentation for the project.
├── notebooks           <- Jupyter notebooks. Includes example implementations.
├── pyproject.toml      <- Project configuration file with package metadata and dependencies.
├── tests               <- Test scripts.
└── class_factory       <- Source code for this project.
    ├── beamer_bot      <- Source code for BeamerBot slide generation.
    ├── ClassFactory.py <- Core class for creating modules (BeamerBot, ConceptWeb, QuizMaker).
    ├── concept_web     <- Source code for concept map generation.
    ├── quiz_maker      <- Source code for quiz generation.
    └── utils           <- Utility functions for OCR, loading documents, etc.
```

---

## How to Use ClassFactory

To get started with ClassFactory, ensure that you have configured your environment correctly (e.g., API keys for LLMs, or a LLM on your machine) and set the paths for your project directories.

### Example Implementation (using `run_classfactory.py`)

```python
from langchain_openai import ChatOpenAI
from pyprojroot.here import here
from pathlib import Path
from class_factory.ClassFactory import ClassFactory

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
    llm=llm,
    project_dir=here(),
    lesson_range=range(17, 21),
    course_name="American Government"
)

```

### Key Modules

#### **BeamerBot**
This module automatically generates LaTeX Beamer slides based on lesson objectives, readings, and prior lessons. It uses an LLM to fill in the slides according to user-specified prompts or default guidance.

```python
beamerbot = factory.create_module("BeamerBot", verbose=True, slide_dir=slideDir)
slides = beamerbot.generate_slides()
beamerbot.save_slides(slides)
```

#### **ConceptWeb**
ConceptWeb builds a visual concept map from lesson materials and readings. It extracts relationships between concepts using LLM-driven text summarization and relationship extraction.

```python
builder = factory.create_module("ConceptWeb", verbose=True, lesson_range=range(19, 21)) # If lesson_range is not specified it will inherit from the factory class
builder.build_concept_map()
```

#### **QuizMaker**
QuizMaker generates quiz questions from the readings and objectives. It ensures diversity in questions by comparing newly generated questions with prior quizzes using embedding similarity checks. It also has the ability to launch an interactive quiz instance via Gradio. If distributed to students, each response will be saved as a unique user_id in your output directory of choice.

```python
quizmaker = factory.create_module("QuizMaker",
                                    verbose=True,
                                    lesson_range=range(19, 21),                # If lesson_range is not specified it will inherit from the factory class
                                    prior_quiz_path = Path(path/to/quiz))      # If using prior_quiz_path, ensure the excel doc passed is of column structure: ['type', 'question', 'A)', 'B)', 'C)', 'D)', 'correct_answer']
quiz = quizmaker.make_a_quiz(difficulty_level=5)  # on a scale of 1-10
quizmaker.save_quiz(quiz)
quizmaker.save_quiz_to_ppt(quiz)
quizmaker.launch_interactive_quiz(quiz_data=quiz, qr_name="quiz_qr_code")
```

---
### **Note on Quality Assurance:**
- Each module in ClassFactory includes automated validation of LLM-generated responses to ensure accuracy, completeness, and consistency.
- The integrated Validator class evaluates every response, automatically forcing the LLM to retry if a response falls below a set quality threshold (limit 3 retries).
- The Validator also adjust prompts dynamically to improve alignment with task requirements if the LLM's initial response fails, ensuring high-quality outputs across all modules.
- See `class_factory/utils/llm_validator` for the validator implementation.
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

You can install the necessary dependencies using the pyproject.toml:

```bash
# If using Poetry (recommended)
poetry install

# If using pip (alternative method)
pip install -r requirements.txt
```

### External dependency prerequisites:
- **LaTeX Compiler**: Required for BeamerBot LaTeX slide generation. Make sure to install a LaTeX distribution like TeX Live or MikTeX. See the options at [The LaTeX Project's Page](https://www.latex-project.org/get/) for the best option for your operating system.


### Optional Dependencies

ClassFactory provides optional dependency groups to extend functionality. These can be installed as needed:

- **OCR**: Dependencies for optical character recognition (OCR) to extract text from images and PDFs:
  ```bash
  poetry install -E ocr
  ```
  Required tools: `pytesseract`, `pillow`, `pdf2image`,`spacy`, `contextualspellcheck`, `img2table`.


#### Required External Installations for OCR

To fully support OCR capabilities, please install the following system dependencies:

1. **Tesseract OCR**
   - `pytesseract` requires Tesseract OCR, an open-source text recognition engine. Follow the installation instructions on [Tesseract OCR's GitHub page](https://github.com/tesseract-ocr/tesseract) for your operating system.

2. **Poppler**
   - `pdf2image` requires Poppler to convert PDF files to images. Visit the [pdf2image GitHub page](https://github.com/Belval/pdf2image) for specific installation instructions.

3. **Spacy Language Model**:
   - OCR operations can often infer incorrect spellings. To fix this and include contextual spell-checking, we need to download the Spacy model:
  ```bash
  python -m spacy download en_core_web_lg
  ```


---

### Folder Structure

ClassFactory assumes a specific folder structure for input and output data:

- Reading Directory: ClassFactory assumes a directory of directories, each with readings for a specific lesson
    - eg
      ```
          readingsDir              <- Directory of Directories
                ├── L1             <- All readings for Lesson 1
                └── L2             <- All readings for Lesson 2
      ...etc
      ```
- Slide directory: Directory containing slides from a prior lesson that BeamerBot will use as context to structure its currently generated lesson. A slide directory is only required for BeamerBot
- Syllabus path: Path to the course syllabus. BeamerBot and ContextWeb will use the current lesson objectives as context to help build either the lesson slides, or the concept map, respectively.
---

### Customization and Extensibility

ClassFactory is designed to be modular. Each module supports custom input and output directories, so outputs can be flexibly stored or processed further. Most development was accomplished using OpenAI's `gpt-4o-mini` but the module supports any user-provided LLM. We have had success using Mistral and LLaMa via `Ollama` for slide and concept map generation, although they both struggled to generate quiz questions consistently. LLaMa was more successful but still unreliable. Some prompt engineering may be required for other models to ensure the model returns the requested JSON-structured output.

### Logging and Debugging

The modules have built-in logging capabilities, with verbosity controlled during initialization. "Verbose=False" sets the logging level to logging.WARNING, otherwise the default is logging.INFO.

---

### Contributing

We welcome contributions! If you'd like to contribute, please fork this repository and submit a pull request. Make sure to include tests for any new functionality and to adhere to the established code structure.
