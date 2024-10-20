"""
ConceptWeb Module

This module defines the `ConceptMapBuilder` class, which automates the extraction of concepts and relationships from lesson materials using a language model (LLM). It visualizes these concepts and their connections through an interactive graph, enabling a deeper understanding of how ideas connect across lessons.

Key functionalities include:

- **Concept Extraction**:
    Extracts key concepts from lesson readings and objectives using the language model, summarizing key themes and topics.

- **Relationship Mapping**:
    Identifies and maps relationships between extracted concepts based on the lesson objectives and reading content.

- **Graph-Based Visualization**:
    Builds a graph where nodes represent concepts and edges represent relationships between concepts. The graph can be visualized interactively as an HTML file, or as a word cloud representing key concepts.

- **Community Detection**:
    Detects communities or clusters of closely related concepts within the graph, providing insight into the thematic structure of the lessons.

- **Intermediate Data Saving**:
    Optionally saves intermediate data such as extracted concepts and relationships as JSON files for further analysis.

Dependencies:

- `langchain_core`: For LLM integration and prompt handling.
- `networkx`: For graph generation and analysis of relationships between concepts.
- `matplotlib` or `plotly`: For generating interactive visualizations and word clouds.
- Custom utilities for loading documents, extracting lesson objectives, and logging.

Usage:

1. **Initialize ConceptMapBuilder**:
    Create an instance of `ConceptMapBuilder` with the project directory, readings directory, and path to your syllabus.

2. **Generate Concept Map**:
    Call `build_concept_map()` to load lesson materials, summarize content, extract relationships, and visualize the concepts.

3. **Visualize and Save**:
    The resulting concept map can be saved as an interactive HTML graph or as a word cloud image.
"""


import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

# parser setup
from langchain_core.output_parsers import JsonOutputParser

# self-made conceptweb functions
from src.concept_web.build_concept_map import build_graph, detect_communities
from src.concept_web.concept_extraction import (
    extract_concepts_from_relationships, extract_relationships,
    process_relationships, summarize_text)
from src.concept_web.prompts import relationship_prompt, summary_prompt
from src.concept_web.visualize_graph import (generate_wordcloud,
                                             visualize_graph_interactive)
from src.utils.load_documents import extract_lesson_objectives, load_lessons
from src.utils.response_parsers import Extracted_Relations
# general utils
from src.utils.tools import logger_setup, reset_loggers

reset_loggers()

# %%


class ConceptMapBuilder:
    """
    A class to generate concept maps from lesson readings and objectives using a language model.

    This class provides an end-to-end workflow to extract, process, and visualize concepts and their relationships
    from lesson materials. It supports loading documents and lesson objectives, summarizing text, extracting relationships,
    building graphs, detecting communities, and creating both interactive visualizations and word clouds.

    Attributes:
        lesson_range (range): The range of lessons to process.
        syllabus_path (Path): Path to the syllabus file containing lesson objectives. Filetypes supported are .docx and .pdf
        readings_dir (Path): Path to the directory containing lesson readings. Filetypes supported are .docx, .txt, and .pdf
        llm (Any): The language model used for text summarization and relationship extraction.
        course_name (str): Name of the course for contextualizing LLM responses.
        recursive (bool): Whether to search for lesson readings recursively in subdirectories.
        concept_list (List[str]): List of unique concepts extracted from the relationships.
        relationship_list (List[Tuple[str, str, str]]): List of relationships between concepts.
        G (nx.Graph): The generated graph of concepts and relationships.
        lesson_objectives (Union[List[str], Dict[str, str]]): User-provided lesson objectives, if any.

    Methods:
        load_lesson_materials():
            Loads the lesson readings and objectives from the specified directories and syllabus.

        summarize_and_extract():
            Summarizes the readings and extracts key relationships using the specified language model.

        build_and_detect_communities():
            Builds a graph from the extracted relationships and detects communities within the graph.

        visualize_graph(output_path: str):
            Generates an interactive graph visualization and saves it as an HTML file.

        generate_wordcloud(output_path: str = None):
            Generates a word cloud of the extracted concepts and optionally saves it as an image file.

        run_pipeline(output_dir: Path):
            Runs the full concept map generation pipeline, from loading materials to saving visualizations.
    """

    def __init__(self, project_dir: Union[str, Path], readings_dir: Union[str, Path], syllabus_path: Union[str, Path],
                 llm, course_name: str, output_dir: Union[str, Path] = None, lesson_range: Union[range, int] = None,
                 recursive: bool = True, lesson_objectives: Union[List[str], Dict[str, str]] = None, verbose: bool = False,
                 save_relationships: bool = False, **kwargs):
        """
        Initializes the ConceptMapBuilder with paths and configurations.

        Args:
            project_dir (Union[str, Path]): The base project directory.
            syllabus_path (Union[str, Path]): The path to the syllabus document (PDF or DOCX).
            llm: The language model instance for summarization and relationship extraction.
            course_name (str): The name of the course (e.g., "American Government").
            lesson_range (range, optional): The range of lesson numbers to load. Defaults to None.
            recursive (bool, optional): Whether to load lessons recursively. Defaults to True.
            save_relationships (bool, optional): Whether to save the generated concepts and relationships to json
        """
        self.project_dir = Path(project_dir)
        self.syllabus_path = Path(syllabus_path)
        self.llm = llm
        self.course_name = course_name
        self.lesson_range = range(lesson_range, lesson_range + 1) if isinstance(lesson_range, int) else lesson_range
        self.recursive = recursive
        self.readings_dir = Path(readings_dir)
        self.data_dir = self.project_dir / "data"
        self.save_relationships = save_relationships
        self.relationship_list = []
        self.concept_list = []
        self.prompts = {'summary': summary_prompt,
                        'relationship': relationship_prompt}
        self.G = None
        self.user_objectives = self.set_user_objectives(lesson_objectives) if lesson_objectives else {}
        log_level = logging.INFO if verbose else logging.WARNING
        self.logger = logger_setup(log_level=log_level)
        self.timestamp = datetime.now().strftime("%Y%m%d")
        if not output_dir:
            rng = [min(self.lesson_range), max(self.lesson_range)]
            self.output_dir = Path(project_dir) / \
                f"reports/ConceptWebOutput/L{rng[0]}_{rng[1]}" if rng[0] != rng[1] else Path(output_dir) / f"L{rng[0]}"
        else:
            rng = [min(self.lesson_range), max(self.lesson_range)]
            self.output_dir = Path(output_dir) / f"L{rng[0]}_{rng[1]}" if rng[0] != rng[1] else Path(output_dir) / f"L{rng[0]}"

        self.kwargs = kwargs

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_user_objectives(self, objectives: Union[List[str], Dict[str, str]]):
        """
        Set the lesson objectives provided by the user.

        If a list is provided, it is converted to a dictionary with keys in the format 'Lesson X',
        where 'X' corresponds to each lesson in the `lesson_range`. If a dictionary is provided, it
        should already be structured with lesson numbers as keys.

        Args:
            objectives (Union[List[str], Dict[str, str]]): The user-provided lesson objectives, either as a list
                (which will be converted to a dictionary) or as a dictionary where keys correspond to the corresponding lesson indicator.

        Raises:
            ValueError: If the length of the objectives list does not match the number of lessons in `lesson_range`.
            TypeError: If `objectives` is not a list or dictionary.
        """
        if isinstance(objectives, list):
            if len(objectives) != len(self.lesson_range):
                raise ValueError("Length of objectives list must match the number of lessons in lesson_range.")
            self.user_objectives = {f'Lesson {i}': obj for i, obj in zip(self.lesson_range, objectives)}
        elif isinstance(objectives, dict):
            if len(objectives) != len(self.lesson_range):
                raise ValueError("Length of objectives list must match the number of lessons in lesson_range.")
            self.user_objectives = objectives
        else:
            raise TypeError("Objectives must be provided as either a list or a dictionary.")

    def load_and_process_lessons(self, summary_prompt: str, relationship_prompt: str):
        """
        Load lesson documents and process them by summarizing the content and extracting key relationships between concepts.

        For each lesson in the specified `lesson_range`, the method:
        1. Loads lesson documents from the specified `readings_dir`.
        2. Extracts lesson objectives from the `syllabus_path`.
        3. Summarizes the lesson readings using the provided language model (LLM).
        4. Extracts relationships between concepts in the readings, based on the lesson objectives.
        5. Extracts unique concepts from the relationships and normalizes them.

        Args:
            summary_prompt (str): The prompt used to guide the LLM in summarizing the lesson readings.
            relationship_prompt (str): The prompt used to guide the LLM in extracting relationships between concepts.
        """
        self.logger.info(f"\nLoading lessons from {self.readings_dir}...")

        # Load documents and lesson objectives
        for lesson_num in self.lesson_range:
            if self.user_objectives:
                self.logger.info('User objectives provided, ignoring provided ')
                lesson_objectives = self.user_objectives.get(f'Lesson {lesson_num}', '')
            else:
                lesson_objectives = extract_lesson_objectives(self.syllabus_path, lesson_num, only_current=True)

            documents = load_lessons(self.readings_dir, lesson_range=range(lesson_num, lesson_num + 1), recursive=self.recursive)

            for document in documents:
                summary = summarize_text(document, prompt=summary_prompt, course_name=self.course_name, llm=self.llm)
                relationships = extract_relationships(summary, lesson_objectives,
                                                      self.course_name,
                                                      llm=self.llm)

                self.relationship_list.extend(relationships)
                concepts = extract_concepts_from_relationships(relationships)
                self.concept_list.extend(concepts)

        # Process relationships to normalize concepts
        self.logger.info("\nExtracting concepts and relations")
        self.relationship_list = process_relationships(self.relationship_list)
        self.concept_list = list(set(self.concept_list))  # Ensure unique concepts

    def save_intermediate_data(self):
        """
        Saves intermediate data, including extracted concepts and relationships, as JSON files for later use.

        This method is triggered when the `save_relationships` flag is set to `True`.

        The following files are saved in the output directory:
        1. `conceptlist_<timestamp>_Lsn_<lesson_range>.json`: Contains the list of unique concepts extracted.
        2. `relationship_list_<timestamp>_Lsn_<lesson_range>.json`: Contains the list of relationships between concepts.

        Files are named with the current timestamp and the lesson range.

        Raises:
            OSError: If there is an issue saving the files.
        """
        with open(self.output_dir / f'conceptlist_{self.timestamp}_Lsn_{self.lesson_range}.json', 'w') as f:
            json.dump(self.concept_list, f)

        with open(self.output_dir / f'relationship_list_{self.timestamp}_Lsn_{self.lesson_range}.json', 'w') as f:
            json.dump(self.relationship_list, f)

    def build_and_visualize_graph(self, method='leiden'):
        """
        Build the concept map as a graph based on the extracted relationships and visualize it in multiple formats.

        Steps:
        1. Build a graph (network) where nodes represent concepts, and edges represent relationships between concepts.
        2. Detect communities within the graph using the specified community detection method ('leiden', 'louvain', or 'spectral').
        3. Generate an interactive HTML file that visualizes the concept map.
        4. Create a word cloud image of the most frequent concepts.

        Args:
            method (str, optional): The community detection method to use. Defaults to 'leiden'. Options are 'leiden', 'louvain', or 'spectral'.

        Outputs:
            An HTML file of the interactive concept map and a PNG word cloud image, both saved in the output directory.

        Raises:
            ValueError: If an unrecognized community detection method is provided.
        """
        self.logger.info("\nBuilding graph...")
        self.G = build_graph(self.relationship_list)

        self.logger.info("\nDetecting communities...")
        # Skip community detection if there's only one lesson
        if len(self.lesson_range) <= 1:
            self.logger.info("\nSingle lesson detected. Skipping community detection.")
            # Assign all nodes to a single community
            for node in self.G.nodes:
                self.G.nodes[node]["community"] = 0  # Assign all nodes to community 0
        else:
            self.logger.info("\nDetecting communities...")
            if method not in ['leiden', 'louvain', 'spectral']:
                raise ValueError("Community detection method not recognized. Please select from 'leiden', 'louvain', or 'spectral'.")
            self.G = detect_communities(self.G, method=method)

        output_html_path = self.output_dir / f"interactive_concept_map_{self.timestamp}_Lsn_{self.lesson_range}.html"
        visualize_graph_interactive(self.G, output_html_path)

        wordcloud_path = self.output_dir / f"concept_wordcloud_{self.timestamp}_Lsn_{self.lesson_range}.png"
        generate_wordcloud(self.concept_list, output_path=wordcloud_path)

    def build_concept_map(self):
        """
        Runs the full pipeline for generating a concept map, from loading lessons to producing visual outputs.

        This method orchestrates the entire process of generating a concept map, including:

        1. **Loading and Processing Lessons:**
           - Lessons and associated readings are loaded based on the provided lesson range and reading directories.
           - If user-provided objectives are available, they will be used; otherwise, the method will extract objectives from the syllabus.

        2. **Summarizing and Extracting Relationships:**
           - Each reading is summarized using the provided language model (LLM).
           - Key relationships between concepts are extracted and processed to normalize concept names.
           - If custom prompts are provided, they will be used for summarization and relationship extraction.

        3. **Building the Concept Map:**
           - Constructs a graph based on the extracted relationships.
           - Detects communities within the graph using the specified community detection method.

        4. **Visualizing the Concept Map:**
           - Generates an interactive HTML visualization of the concept map.
           - Creates a word cloud image representing the most frequent concepts.

        Outputs:
            - An interactive HTML file of the concept map is saved to the output directory.
            - A PNG word cloud image is generated and saved to the output directory.

        Note:
            This method uses defaults set during class initialization unless overridden by the provided arguments.

        Raises:
            ValueError: If any part of the process encounters invalid or inconsistent data.
        """
        summary_prompt = self.kwargs.get('summary_prompt', self.prompts['summary'])
        relationship_prompt = self.kwargs.get('relationship_prompt', self.prompts['relationship'])
        method = self.kwargs.get('method', 'leiden')

        self.load_and_process_lessons(summary_prompt=summary_prompt, relationship_prompt=relationship_prompt)
        if self.save_relationships:
            self.save_intermediate_data()
        self.build_and_visualize_graph(method=method)


if __name__ == "__main__":
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    # env setup
    from pyprojroot.here import here
    load_dotenv()

    # llm chain setup
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI

    # Path definitions
    readingDir = Path(os.getenv('readingsDir'))
    syllabus_path = Path(os.getenv('syllabus_path'))
    # pdf_syllabus_path = Path(os.getenv('pdf_syllabus_path'))

    projectDir = here()

    # Example usage
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv('openai_key'),
        organization=os.getenv('openai_org'),
    )
    # llm = Ollama(
    #     model="llama3.1",
    #     temperature=0.1,

    #     )

    builder = ConceptMapBuilder(
        readings_dir=readingDir,
        project_dir=projectDir,
        syllabus_path=syllabus_path,
        llm=llm,
        course_name="American Politics",
        lesson_range=range(19, 21),
        output_dir=readingDir/"L20",
        recursive=True,
        verbose=True
    )

    builder.build_concept_map()
