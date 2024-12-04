"""
This script processes lesson readings and objectives to extract key concepts and their relationships,
which are then prepared for export to a graph or other forms of visualization.

The script performs the following steps:
    1. **Summarize Text**: Summarizes the provided lesson readings using a specified prompt and language model.
    2. **Extract Relationships**: Extracts key concepts and relationships between them from the summarized text,
       guided by lesson objectives and prompts.
    3. **Normalize and Process Relationships**: Normalizes concept names, consolidates similar concepts,
       and processes the relationships to create a structured output.
    4. **Process Output**: Processes the relationships to ensure consistency in concept naming and
       prepares the data for further use (e.g., in graph creation).

Dependencies:
    - **Language Model**: Requires a language model (such as OpenAI's GPT) for summarizing readings and extracting relationships.
    - **JSON**: Used for parsing and managing data extracted from the language model.
    - **Inflect**: Utilized for normalizing concept names, including singularizing nouns.
    - **Regex**: Applied for text processing and pattern matching in various normalization and extraction steps.

This script is intended to serve as a modular component in a larger pipeline, where extracted relationships and concepts can be further analyzed, visualized, or exported for educational purposes.
"""

# base libraries
import json
# logger setup
import logging
import os
from pathlib import Path
from typing import Any, List, Set, Tuple

import inflect
# env setup
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from class_factory.concept_web.prompts import (
    no_objective_relationship_prompt, relationship_prompt, summary_prompt)
from class_factory.utils.llm_validator import Validator
from class_factory.utils.response_parsers import (Extracted_Relations,
                                                  ValidatorResponse)
from class_factory.utils.tools import logger_setup, retry_on_json_decode_error

# logging.basicConfig(
#     level=logging.INFO,  # Set your desired level
#     format='%(name)s - %(levelname)s - %(message)s'
# )

# %%


def summarize_text(text: str, prompt: ChatPromptTemplate, course_name: str, llm: Any, parser=StrOutputParser(), verbose=False) -> str:
    """
    Summarize the provided text using the specified prompt and objectives.

    Args:
        text (str): The text to be summarized.
        prompt (str): The prompt template to guide the summarization.
        llm (Any): The language model instance to use for generating the summary.
        parser (StrOutputParser): The parser to handle the output.

    Returns:
        str: The summary generated by the language model.
    """
    log_level = logging.INFO if verbose else logging.ERROR
    logger = logger_setup(log_level=log_level)

    # summary_template = PromptTemplate.from_template(prompt)
    chain = prompt | llm | parser
    summary = chain.invoke({'course_name': course_name,
                            'text': text})
    logger.info(f"Example summary:\n{summary}")

    return summary


@retry_on_json_decode_error()
def extract_relationships(text: str, objectives: str, course_name: str,
                          llm: Any, verbose=False, logger=None) -> List[Tuple[str, str, str]]:
    """
    Extract key concepts and their relationships from the provided text.

    Args:
        text (str): The summarized text.
        objectives (str): Lesson objectives to guide the relationship extraction.
        course_name (str): The name of the course (e.g., "American Government", "International Relations").
        llm (Any): The language model to use for generating responses.
        verbose (bool): Whether to use verbose logging.
        logger (logging.Logger, optional): The logger to use for logging.

    Returns:
        List[Tuple[str, str, str]]: A list of tuples representing the relationships between concepts.
    """
    log_level = logging.INFO if verbose else logging.WARNING
    logger = logger or logging.getLogger(__name__)
    logger.setLevel(log_level)

    parser = JsonOutputParser(pydantic_object=Extracted_Relations)
    val_parser = JsonOutputParser(pydantic_object=ValidatorResponse)

    if objectives:
        selected_prompt = relationship_prompt
    else:
        selected_prompt = no_objective_relationship_prompt
        objectives = "Not provided."

    additional_guidance = ""
    # combined_template = PromptTemplate.from_template(selected_prompt)
    chain = selected_prompt | llm | parser

    logger.debug(f"""Querying with:\n{selected_prompt.format(course_name=course_name,
                                                 objectives=objectives,
                                                 text="placeholder",
                                                 additional_guidance="")}""")

    validator = Validator(llm=llm, parser=val_parser, log_level=log_level)
    retries, max_retries = 0, 3
    valid = False

    while not valid and retries < max_retries:
        response = chain.invoke({'course_name': course_name,
                                 'objectives': objectives,
                                 'text': text,
                                 'additional_guidance': additional_guidance})

        # Clean and parse the JSON output
        if isinstance(response, str):
            response_cleaned = response.replace("```json", "").replace("```", "")
            data = json.loads(response_cleaned)  # This may raise JSONDecodeError
        else:
            data = response

        # Verify that data is a dict
        if not isinstance(data, dict):
            logger.error("Parsed data is not a dictionary.")
            raise ValueError("Parsed data is not a dictionary.")

        # Validate responses
        # escape curly braces for langchain invoke with double curlies
        response_str = json.dumps(response).replace("{", "{{").replace("}", "}}")
        validation_prompt = selected_prompt.format(course_name=course_name,
                                                   objectives=objectives,
                                                   text=text,
                                                   additional_guidance=additional_guidance
                                                   ).replace("{", "{{").replace("}", "}}")

        val_response = validator.validate(task_description=validation_prompt,
                                          generated_response=response_str,
                                          min_eval_score=8)

        logger.info(f"validation output: {val_response}")
        if int(val_response['status']) == 1:
            valid = True
        else:
            retries += 1
            additional_guidance = val_response.get("additional_guidance", "")
            logger.warning(f"Validation failed on attempt {retries}. Reason: {val_response['reasoning']}")

    if valid:
        logger.info("Validation succeeded.")
    else:
        raise ValueError("Validation failed after max retries. Ensure correct prompt and input data. Consider use of a different LLM.")

    # Extract concepts and relationships
    relationships = [tuple(relationship) for relationship in data["relationships"]]
    return relationships


def extract_concepts_from_relationships(relationships: List[Tuple[str, str, str]]) -> List[str]:
    """
    Extract unique concepts from the list of relationships.

    Args:
        relationships (list): List of tuples representing relationships between concepts.

    Returns:
        list: A list of unique concepts.
    """
    concepts = set()  # Use a set to avoid duplicates
    for concept1, _, concept2 in relationships:
        concepts.add(concept1)
        concepts.add(concept2)
    return list(concepts)


def normalize_concept(concept: str) -> str:
    """
    Normalize a concept by converting it to lowercase, replacing spaces with underscores, and converting plural forms to singular.

    Args:
        concept (str): The concept to normalize.

    Returns:
        str: The normalized concept.
    """
    p = inflect.engine()

    # Normalize case, remove extra spaces, and split on spaces and underscores
    words = concept.lower().strip().replace('_', ' ').split()

    normalized_words = [
        p.singular_noun(word) if word != 'is' and p.singular_noun(word) else word
        for word in words
    ]
    return "_".join(normalized_words)


def jaccard_similarity(concept1: str, concept2: str, threshold: float = 0.85) -> bool:
    """
    Calculate the Jaccard similarity between two concepts.

    Args:
        concept1 (str): The first concept.
        concept2 (str): The second concept.
        threshold (float): The similarity threshold.

    Returns:
        bool: True if the similarity exceeds the threshold, False otherwise.
    """
    set1 = set(concept1)
    set2 = set(concept2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity >= threshold


def replace_similar_concepts(existing_concepts: Set[str], new_concept: str, threshold: float = 0.85) -> str:
    """
    Replace a new concept with an existing similar concept if found.

    Args:
        existing_concepts (set): Set of existing concepts.
        new_concept (str): The new concept to check.

    Returns:
        str: The existing concept if a match is found, otherwise the new concept.
    """
    for existing_concept in existing_concepts:
        # If concepts are too similar, consolidate naming
        if jaccard_similarity(existing_concept, new_concept, threshold):
            return existing_concept
    return new_concept


def process_relationships(relationships: List[Tuple[str, str, str]], threshold: float = 0.85) -> List[Tuple[str, str, str]]:
    """
    Process and normalize relationships by consolidating similar concepts.

    Args:
        relationships (list): List of tuples representing relationships between concepts.

    Returns:
        list: Processed relationships with normalized concepts.
    """
    # Initialize a set to keep track of all unique concepts
    unique_concepts = set()
    processed_relationships = []

    if not isinstance(relationships[0], tuple):
        relationships = [tuple(relation) for relation in relationships]

    for c1, relationship, c2 in relationships:
        # Normalize concepts
        clean_concept1 = normalize_concept(c1)
        clean_concept2 = normalize_concept(c2)
        clean_relation = normalize_concept(relationship)

        # Replace similar concepts with existing ones
        concept1 = replace_similar_concepts(unique_concepts, clean_concept1, threshold)
        concept2 = replace_similar_concepts(unique_concepts, clean_concept2, threshold)

        # Add concepts to the unique set
        unique_concepts.add(concept1)
        unique_concepts.add(concept2)

        # Add the relationship to the processed list
        processed_relationships.append((concept1, clean_relation, concept2))

    return processed_relationships


if __name__ == "__main__":
    # llm chain setup
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from pyprojroot.here import here

    # self-defined utils
    from class_factory.utils.load_documents import LessonLoader
    user_home = Path.home()
    load_dotenv()

    OPENAI_KEY = os.getenv('openai_key')
    OPENAI_ORG = os.getenv('openai_org')

    # Path definitions
    readingDir = user_home / os.getenv('readingsDir')
    slideDir = user_home / os.getenv('slideDir')
    syllabus_path = user_home / os.getenv('syllabus_path')
    pdf_syllabus_path = user_home / os.getenv('pdf_syllabus_path')

    projectDir = here()

    parser = JsonOutputParser(pydantic_object=Extracted_Relations)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_KEY,
        organization=OPENAI_ORG,
    )

    # llm = Ollama(
    #     model="llama3.1",
    #     temperature=0.5,

    #     )

    relationship_list = []
    conceptlist = []

    loader = LessonLoader(syllabus_path=syllabus_path,
                          reading_dir=readingDir,
                          slide_dir=None)

    # Load documents and lesson objectives
    for lesson_num in range(1, 3):
        print(f"Lesson {lesson_num}")
        lesson_objectives = loader.extract_lesson_objectives(current_lesson=lesson_num)
        documents = loader.load_lessons(lesson_number_or_range=range(lesson_num, lesson_num + 1))

        if not documents:
            continue

        for lsn, readings in documents.items():
            for reading in readings:
                summary = summarize_text(reading,
                                         prompt=summary_prompt,
                                         course_name="American government",
                                         llm=llm,
                                         verbose=False)
                # print(summary)
                relationships = extract_relationships(summary,
                                                      lesson_objectives,
                                                      course_name="American government",
                                                      llm=llm,
                                                      verbose=True)
                print(relationships)
                relationship_list.extend(relationships)

                concepts = extract_concepts_from_relationships(relationships)
                conceptlist.extend(concepts)

        processed_relationships = process_relationships(relationship_list)
