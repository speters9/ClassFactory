Configuration Guide
===================

Configuration File Structure
-----------------------------

ClassFactory uses a YAML configuration file (``class_config.yaml``) to manage course settings. Each course is configured as a separate section:

.. code-block:: yaml

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

Configuration Parameters
------------------------

Course Settings
~~~~~~~~~~~~~~~

* **course_title**: Human-readable name for the course
* **syllabus_path**: Path to the course syllabus document
* **reading_dir**: Directory containing lesson readings
* **slideDir**: Directory for slide templates (BeamerBot only)
* **is_tabular_syllabus**: Boolean indicating if syllabus uses table format

Lesson Objectives
~~~~~~~~~~~~~~~~~

The ``lesson_objectives`` section allows you to manually specify learning objectives for each lesson:

.. code-block:: yaml

    lesson_objectives:
      '1': "Understand basic concepts and terminology"
      '2': "Analyze historical examples and case studies"
      '3': "Apply theoretical frameworks to real-world scenarios"

API Key Setup
-------------

Store your API keys securely in a ``.env`` file in the project root:

.. code-block:: bash

    # .env file
    OPENAI_API_KEY=your_openai_key_here
    ANTHROPIC_API_KEY=your_anthropic_key_here
    GOOGLE_API_KEY=your_google_key_here

Directory Structure
-------------------

ClassFactory expects a specific folder structure:

Reading Directory
~~~~~~~~~~~~~~~~~

Organize readings by lesson::

    readings/
    ├── L1/                 # All readings for Lesson 1
    │   ├── reading1.pdf
    │   └── reading2.docx
    ├── L2/                 # All readings for Lesson 2
    │   └── reading3.pdf
    └── ...

Slide Directory
~~~~~~~~~~~~~~~

For BeamerBot, organize slides by lesson (optional)::

    slides/
    ├── L1/
    │   └── lesson1_slides.tex
    ├── L2/
    │   └── lesson2_slides.tex
    └── ...

Getting Started
---------------

1. **Copy the example configuration**::

    cp class_config_example.yaml class_config.yaml

2. **Edit with your course details**: Update paths, course information, and objectives

3. **Create the .env file**: Add your API keys

4. **Use the notebook runner**: Open ``notebooks/run_classfactory.py`` to run modules interactively
