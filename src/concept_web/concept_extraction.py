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

from src.concept_web.prompts import (no_objective_relationship_prompt,
                                     relationship_prompt, summary_prompt)
from src.utils.response_parsers import Extracted_Relations
from src.utils.tools import logger_setup, retry_on_json_decode_error

# logging.basicConfig(
#     level=logging.INFO,  # Set your desired level
#     format='%(name)s - %(levelname)s - %(message)s'
# )

logging.getLogger('httpx').setLevel(logging.WARNING)

# %%


def summarize_text(text: str, prompt: str, course_name: str, llm: Any, parser=StrOutputParser(), verbose=False) -> str:
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

    summary_template = PromptTemplate.from_template(prompt)
    chain = summary_template | llm | parser
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
    log_level = logging.INFO if verbose else logging.ERROR
    logger = logger or logging.getLogger(__name__)
    logger.setLevel(log_level)

    parser = JsonOutputParser(pydantic_object=Extracted_Relations)

    if objectives:
        selected_prompt = relationship_prompt
    else:
        selected_prompt = no_objective_relationship_prompt
        objectives = "Not provided."

    combined_template = PromptTemplate.from_template(selected_prompt)
    chain = combined_template | llm | parser

    logger.info(f"Querying with:\n{selected_prompt}")

    response = chain.invoke({'course_name': course_name,
                             'objectives': objectives,
                             'text': text})

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

    # self-defined utils
    from src.utils.load_documents import (extract_lesson_objectives,
                                          load_readings)
    load_dotenv()

    OPENAI_KEY = os.getenv('openai_key')
    OPENAI_ORG = os.getenv('openai_org')

    # Path definitions
    readingDir = Path(os.getenv('readingsDir'))
    slideDir = Path(os.getenv('slideDir'))
    syllabus_path = Path(os.getenv('syllabus_path'))
    pdf_syllabus_path = Path(os.getenv('pdf_syllabus_path'))

    projectDir = Path(os.getenv('projectDir'))

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

    for lsn in range(1, 3):
        print(f"Extracting Lesson {lsn}")
        lsn_summaries = []
        readings = []
        objectives = ['']
        inputDir = readingDir / f'L{lsn}/'
        # load readings from the lesson folder
        if os.path.exists(inputDir):
            for file in inputDir.iterdir():
                if file.suffix in ['.pdf', '.docx', '.txt']:
                    readings_text = load_readings(file)
                    readings.append(readings_text)

        if not readings:
            continue

        lsn_objectives = extract_lesson_objectives(syllabus_path,
                                                   lsn,
                                                   only_current=True)

        for reading in readings:
            summary = summarize_text(reading,
                                     prompt=summary_prompt,
                                     course_name="American government",
                                     llm=llm,
                                     verbose=False)
            # print(summary)
            relationships = extract_relationships(summary,
                                                  lsn_objectives,
                                                  course_name="American government",
                                                  llm=llm,
                                                  verbose=False)
            print(relationships)
            relationship_list.extend(relationships)

            concepts = extract_concepts_from_relationships(relationships)
            conceptlist.extend(concepts)

        processed_relationships = process_relationships(relationship_list)