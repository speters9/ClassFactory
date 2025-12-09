Installation Guide
==================

System Requirements
-------------------

* Python 3.11 or higher
* Windows, macOS, or Linux
* LaTeX distribution (for BeamerBot slide generation)

Installation Methods
--------------------

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

ClassFactory is designed to work with ``uv``, a fast Python package manager::

    # Clone the repository
    git clone https://github.com/speters9/ClassFactory.git
    cd ClassFactory

    # Install dependencies
    uv sync

    # Install development dependencies (optional)
    uv sync --group dev

Using pip
~~~~~~~~~

Alternatively, you can use traditional pip installation::

    # Clone the repository
    git clone https://github.com/speters9/ClassFactory.git
    cd ClassFactory

    # Create virtual environment (recommended)
    python -m venv classfactory_env

    # Activate virtual environment
    # Windows:
    classfactory_env\Scripts\activate
    # macOS/Linux:
    source classfactory_env/bin/activate

    # Install dependencies
    pip install -r requirements.txt

External Dependencies
--------------------

LaTeX Distribution
~~~~~~~~~~~~~~~~~~

Required for BeamerBot slide generation:

* **Windows**: MikTeX or TeX Live
* **macOS**: MacTeX or TeX Live
* **Linux**: TeX Live

Download from `The LaTeX Project <https://www.latex-project.org/get/>`_.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~

OCR Support
^^^^^^^^^^^

For processing scanned documents, install:

* **Tesseract OCR**: Follow instructions at `Tesseract GitHub <https://github.com/tesseract-ocr/tesseract>`_
* **Poppler**: Required for PDF to image conversion

Verification
------------

Test your installation::

    python -c "from class_factory.ClassFactory import ClassFactory; print('Installation successful!')"

If you see "Installation successful!", you're ready to use ClassFactory!
