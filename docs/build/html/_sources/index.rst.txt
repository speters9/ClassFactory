.. ClassFactory documentation master file, created by
   sphinx-quickstart on Tue Dec  9 10:46:01 2025.

ClassFactory Documentation
==========================

**AI-powered tools for instructional product generation**

ClassFactory is a modular toolkit designed to automate various aspects of lesson and course material generation leveraging the generative capabilities of large language models (LLMs). It offers functionality to create interactive learning resources, including LaTeX Beamer slides, concept maps, and quizzes, all structured around a specified syllabus or lesson plan.

Key Features
------------

* **BeamerBot** - Automated LaTeX Beamer slide generation
* **ConceptWeb** - Interactive concept maps and knowledge graphs
* **QuizMaker** - Quiz generation with similarity checking and interactive features
* **LLM Integration** - Support for OpenAI, Anthropic, Google, and Ollama
* **Quality Assurance** - Built-in validation and retry mechanisms

Quick Start
-----------

1. **Install ClassFactory**::

    # Using uv (recommended)
    uv sync

    # Or using pip
    pip install -r requirements.txt

2. **Configure your course**: Copy ``class_config_example.yaml`` to ``class_config.yaml`` and customize

3. **Run ClassFactory**::

    python run_classfactory.py

The interactive interface will guide you through module selection and configuration.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Quick Start <self>
   modules
   installation
   configuration
   examples

API Documentation
-----------------

.. toctree::
   :maxdepth: 3
   :caption: API Reference:
