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

## Quick Start

1. **Configure your course**: Edit `class_config.yaml` with your course details
2. **Store API keys in your `.env` file**: Create the file and save in the project's root directory
3. **Run ClassFactory**: Use the interactive CLI to generate content

```bash
python run_classfactory.py
```

The interactive interface will guide you through:
- Selecting which module to run (BeamerBot, ConceptWeb, or QuizMaker)
- Choosing lesson ranges for multi-lesson modules
- Configuring LLM settings

### Configuration File Structure

ClassFactory uses a YAML configuration file (`class_config.yaml`) to manage course settings. Each course is configured as a separate section:

```yaml
PS302:
  course_title: "American Foreign Policy"
  syllabus_path: "path/to/syllabus.docx"
  reading_dir: "path/to/readings"
  slideDir: "path/to/slides"
  is_tabular_syllabus: True
  lesson_objectives:
    '1': "Your lesson 1 objectives here"
    '2': "Your lesson 2 objectives here"

PS211:
  course_title: "American Government"
  syllabus_path: "path/to/syllabus.docx"
  reading_dir: "path/to/readings"
  slideDir: "path/to/slides"
  is_tabular_syllabus: True
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

## Installation

After cloning this repository:

```bash
# Option 1: Using uv (recommended - faster and handles pyproject.toml)
uv sync

# Option 2: Using pip with requirements.txt
pip install -r requirements.txt

# Copy and customize the configuration file
cp class_config_example.yaml class_config.yaml
# Edit class_config.yaml with your course details

```

### API Key Setup

Store your API keys securely in a `.env` file in the project root:

```bash
# .env file
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

### External dependency prerequisites:
- **LaTeX Compiler**: Required for BeamerBot LaTeX slide generation. Make sure to install a LaTeX distribution like TeX Live or MikTeX. See the options at [The LaTeX Project's Page](https://www.latex-project.org/get/) for the best option for your operating system.


### Optional Dependencies

ClassFactory provides optional dependency groups to extend functionality. These can be installed as needed:

- **OCR**: Dependencies for optical character recognition (OCR) to extract text from images and PDFs:
  ```bash
  uv sync --group ocr
  ```
  Required packages: `pytesseract`, `pillow`, `pdf2image`.


#### Required External Installations for OCR

To fully support OCR capabilities, please install the following system dependencies:

1. **Tesseract OCR**
   - `pytesseract` requires Tesseract OCR, an open-source text recognition engine. Follow the installation instructions on [Tesseract OCR's GitHub page](https://github.com/tesseract-ocr/tesseract) for your operating system.

2. **Poppler**
   - `pdf2image` requires Poppler to convert PDF files to images. Visit the [pdf2image GitHub page](https://github.com/Belval/pdf2image) for specific installation instructions.


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
- Syllabus path: Path to the course syllabus. BeamerBot and ContextWeb will use the current lesson objectives (if they exist) as context to help build either the lesson slides, or the concept map, respectively.
---

### Customization and Extensibility

ClassFactory is designed to be modular. Each module supports custom input and output directories, so outputs can be flexibly stored or processed further. Most development was accomplished using OpenAI's `gpt-4o-mini` but the module supports any user-provided LLM. We have had success using Mistral and LLaMa via `Ollama` for slide and concept map generation, although they both struggled to generate quiz questions consistently. LLaMa was more successful but still unreliable. Some prompt engineering may be required for other models to ensure the model returns the requested JSON-structured output.

### Logging and Debugging

The modules have built-in logging capabilities, with verbosity controlled during initialization. "Verbose=False" sets the logging level to logging.WARNING, otherwise the default is logging.INFO.

---

### Contributing

We welcome contributions! If you'd like to contribute, please fork this repository and submit a pull request. Make sure to include tests for any new functionality and to adhere to the established code structure.
