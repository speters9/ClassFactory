Examples and Use Cases
======================

This section provides practical examples of using ClassFactory for different educational scenarios.

Basic Usage Examples
--------------------

Generating Slides for a Single Lesson
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from class_factory.ClassFactory import ClassFactory
    from class_factory.utils.tools import get_llm
    from pathlib import Path

    # Initialize LLM
    llm = get_llm("openai")

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
    from class_factory.utils.tools import get_llm

    llm = get_llm("openai")

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

ClassFactory provides a convenient ``get_llm()`` helper function to initialize different LLM providers:

.. code-block:: python

    from class_factory.utils.tools import get_llm

    # OpenAI (uses environment variable for API key)
    llm = get_llm("openai")

    # Anthropic Claude (uses environment variable for API key)
    llm = get_llm("anthropic")

    # Google Gemini (uses environment variable for API key)
    llm = get_llm("gemini")

    # Local models with Ollama (no API key needed)
    llm = get_llm("ollama")

    # You can also pass API keys directly
    llm = get_llm("openai", openai_key="your-api-key")
    llm = get_llm("anthropic", anthropic_key="your-api-key")
    llm = get_llm("gemini", gemini_key="your-api-key")

The function automatically loads API keys from environment variables if not provided directly.

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

Publishing Generated Content
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After generating slides with BeamerBot, they will be saved in the default ClassFactoryOutput/BeamerBot/ directory. After you have modified them to your liking, you can publish them to your master slides directory:

If you want to publish directly to your maser directory, you can just add the output directory as an argument to your ``save_slides()`` call: 
.. code-block:: python

    # Generate and save slides
    beamerbot = factory.create_module("BeamerBot")
    slides = beamerbot.generate_slides()
    beamerbot.save_slides(slides)
    
    # Publish to your master slides folder
    published_path = beamerbot.publish_slides(dest_dir=Path("/path/to/master/slides"))
    # Output: Published L5.tex to /path/to/master/slides/L5.tex

The ``publish_slides()`` method automatically:

* Detects the correct file extension (.tex for LaTeX, .pptx for PowerPoint)
* Creates the destination directory if it doesn't exist
* Copies the file from the output directory to your master directory
* Returns the path to the published file
