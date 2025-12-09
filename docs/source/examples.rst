Examples and Use Cases
======================

This section provides practical examples of using ClassFactory for different educational scenarios.

Basic Usage Examples
--------------------

Generating Slides for a Single Lesson
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from class_factory.ClassFactory import ClassFactory
    from langchain_openai import ChatOpenAI
    from pathlib import Path

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Create factory instance
    factory = ClassFactory(
        lesson_no=5,
        course_config="PS302",  # References config file
        llm=llm
    )

    # Generate slides
    beamerbot = factory.create_module("BeamerBot")
    slides = beamerbot.generate_slides()
    beamerbot.save_slides(slides)

Creating a Concept Map
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create concept web for multiple lessons
    concept_web = factory.create_module(
        "ConceptWeb",
        lesson_range=range(1, 6)  # Lessons 1-5
    )

    concept_web.build_concept_map()

Quiz Generation with Prior Quiz Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path

    # Generate quiz avoiding duplicates
    quiz_maker = factory.create_module(
        "QuizMaker",
        lesson_range=range(3, 5),
        prior_quiz_path=Path("previous_quiz.xlsx")
    )

    quiz = quiz_maker.make_a_quiz(difficulty_level=7)
    quiz_maker.save_quiz(quiz)

    # Launch interactive version
    quiz_maker.launch_interactive_quiz(quiz)

Advanced Workflows
------------------

Multi-Lesson Course Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from class_factory.ClassFactory import ClassFactory
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Prepare materials for lessons 10-15
    lesson_range = range(10, 16)

    for lesson_no in lesson_range:
        factory = ClassFactory(
            lesson_no=lesson_no,
            course_config="PS302",
            llm=llm
        )

        # Generate slides
        beamerbot = factory.create_module("BeamerBot")
        slides = beamerbot.generate_slides()
        beamerbot.save_slides(slides)

        # Generate quiz
        quiz_maker = factory.create_module("QuizMaker")
        quiz = quiz_maker.make_a_quiz(difficulty_level=5)
        quiz_maker.save_quiz(quiz)

    # Create concept map for entire unit
    unit_factory = ClassFactory(
        lesson_no=15,
        course_config="PS302",
        llm=llm
    )

    concept_web = unit_factory.create_module(
        "ConceptWeb",
        lesson_range=lesson_range
    )
    concept_web.build_concept_map()

Using Different LLM Providers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenAI
^^^^^^

.. code-block:: python

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key="your-api-key"
    )

Anthropic Claude
^^^^^^^^^^^^^^^

.. code-block:: python

    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.3,
        api_key="your-api-key"
    )

Google Gemini
^^^^^^^^^^^^^

.. code-block:: python

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key="your-api-key"
    )

Local Models with Ollama
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from langchain_community.llms import Ollama

    llm = Ollama(
        model="mistral",
        temperature=0.3
    )

Interactive CLI Usage
~~~~~~~~~~~~~~~~~~~~~~

The simplest way to use ClassFactory is through the interactive CLI:

.. code-block:: bash

    python run_classfactory.py

This will guide you through:

1. **Course Selection**: Choose from configured courses
2. **Module Selection**: Pick BeamerBot, ConceptWeb, or QuizMaker
3. **Lesson Configuration**: Set lesson numbers and ranges
4. **LLM Configuration**: Choose and configure your language model

The CLI handles all the complexity of module initialization and configuration.

Best Practices
--------------

File Organization
~~~~~~~~~~~~~~~~

* Keep readings organized by lesson in separate directories
* Use descriptive filenames for easy identification
* Maintain consistent naming conventions across lessons

Quality Control
~~~~~~~~~~~~~~

* Review generated content before using in class
* Use the built-in validation features
* Test quizzes with a small group before wide deployment

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

* Use faster models (like gpt-4o-mini) for initial drafts
* Save API costs by generating multiple lessons in batch
* Cache concept maps for reuse across similar courses

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~

* **Import Errors**: Ensure all dependencies are installed with ``uv sync``
* **API Failures**: Check your .env file has valid API keys
* **File Not Found**: Verify all paths in your configuration are correct
* **LaTeX Errors**: Ensure you have a working LaTeX distribution installed

For more help, check the logs (ClassFactory provides detailed logging) or refer to the API documentation.
