"""
**ConceptWeb Module**
-----------------------

The `ConceptWeb` module provides tools to automatically extract, analyze, and visualize key concepts from lesson materials, helping to identify connections across topics and lessons. Central to this module is the `ConceptMapBuilder` class, which leverages a language model (LLM) to identify and structure important ideas and relationships from lesson readings and objectives into a graph-based representation.

Key functionalities of the module include:

- **Concept Extraction**:
    - Identifies key concepts from lesson readings and objectives using an LLM.
    - Summarizes and highlights main themes from each lesson's content.

- **Relationship Mapping**:
    - Extracts and maps relationships between identified concepts based on lesson objectives and content.
    - Facilitates understanding of how topics interrelate within and across lessons.

- **Graph-Based Visualization**:
    - Constructs a concept map in which nodes represent concepts and edges represent relationships.
    - Generates both interactive graph-based visualizations (HTML) and word clouds for key concepts.

- **Community Detection**:
    - Groups closely related concepts into thematic clusters.
    - Helps identify broader themes or subtopics within the lesson materials.

- **Data Saving**:
    - Optionally saves intermediate data (concepts and relationships) as JSON files for further review or analysis.

Dependencies
~~~~~~~~~~~~~

This module depends on:

- `langchain_core`: For LLM-based extraction and summarization tasks.
- `networkx`: For graph generation and analysis of concept relationships.
- `matplotlib` or `plotly`: For creating visualizations and word clouds.
- Custom utilities for loading documents, extracting objectives, and handling logging.

Usage Overview
~~~~~~~~~~~~~~

1. **Initialize ConceptMapBuilder**:
   - Instantiate `ConceptMapBuilder` with paths to project directories, reading materials, and the syllabus file.

2. **Generate the Concept Map**:
   - Use `build_concept_map()` to process lesson materials, extract and summarize concepts, map relationships, and generate visualizations.

3. **Save and Review**:
   - The generated concept map can be saved as an interactive HTML file or as a static word cloud for easier review and analysis.

Example
~~~~~~~~

.. code-block:: python

    from class_factory.concept_web.ConceptMapBuilder import ConceptMapBuilder
    from class_factory.utils.load_documents import LessonLoader
    from langchain_openai import ChatOpenAI

    # Set up paths and initialize components
    syllabus_path = Path("/path/to/syllabus.docx")
    reading_dir = Path("/path/to/lesson/readings")
    project_dir = Path("/path/to/project")
    llm = ChatOpenAI(api_key="your_api_key")

    # Initialize the lesson loader and concept map builder
    lesson_loader = LessonLoader(syllabus_path=syllabus_path, reading_dir=reading_dir, project_dir=project_dir)
    concept_map_builder = ConceptMapBuilder(
        lesson_no=1,
        lesson_loader=lesson_loader,
        llm=llm,
        course_name="Sample Course",
        lesson_range=range(1, 5)
    )

    # Build and visualize the concept map
    concept_map_builder.build_concept_map()


"""

#%%
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# parser setup
from langchain_core.output_parsers import JsonOutputParser

# self-made conceptweb functions
from class_factory.concept_web.build_concept_map import (build_graph,
                                                         detect_communities)
from class_factory.concept_web.concept_extraction import (
    extract_concepts_from_relationships, extract_relationships,
    process_relationships, summarize_text)
from class_factory.concept_web.prompts import (relationship_prompt,
                                               summary_prompt)
from class_factory.concept_web.visualize_graph import (
     visualize_graph_interactive)
from class_factory.utils.base_model import BaseModel
from class_factory.utils.load_documents import LessonLoader

# %%


class ConceptMapBuilder(BaseModel):
    """
    Generate concept maps (a form of knowledge graph) from lesson materials, using a language model (LLM) to summarize content,
    extract relationships, and visualize concepts in a structured graph format.

    This class provides end-to-end functionality for concept map creation, including loading readings,
    summarizing content, extracting concept relationships, constructing graphs, and generating
    interactive and visual outputs like word clouds.

    Attributes:
        lesson_no (int): Current lesson number being processed.
        lesson_loader (LessonLoader): Loader instance for handling lesson materials.
        llm (Any): Language model instance for summarization and relationship extraction.
        course_name (str): Course name, used as context in LLM prompts.
        output_dir (Path): Directory for saving generated outputs.
        lesson_range (range): Range of lessons to process.
        save_relationships (bool): Whether to save extracted relationships to JSON.
        relationship_list (List[Tuple[str, str, str]]): List of concept relationships.
        concept_list (List[str]): List of unique concepts extracted.
        prompts (Dict[str, str]): Dictionary of prompts for LLM tasks.
        verbose (bool): Whether to enable verbose logging.
        G (Optional[nx.Graph]): Generated concept graph.
        user_objectives (Dict[str, str]): User-defined lesson objectives.

    Methods:
        load_and_process_lessons(threshold: float = 0.995):
            Loads lesson materials, summarizes content, and extracts relationships between concepts.

        build_concept_map(directed: bool = False, concept_similarity_threshold: float = 0.995,
                         dark_mode: bool = True, lesson_objectives: Optional[Dict[str, str]] = None):
            Runs the concept map generation pipeline and outputs visualizations.
    """

    def __init__(self, lesson_no: int, lesson_loader: LessonLoader, llm, course_name: str,
                 output_dir: Union[str, Path] = None, lesson_range: Union[range, int] = None,
                 lesson_objectives: Union[List[str], Dict[str, str]] = None,
                 verbose: bool = False, save_relationships: bool = False, **kwargs):

        # Initialize BaseModel with shared attributes
        super().__init__(lesson_no=lesson_no, course_name=course_name, lesson_loader=lesson_loader,
                         output_dir=output_dir, verbose=verbose)
        """
        Initialize the ConceptMapBuilder with paths and configurations for concept map generation.

        Args:
            project_dir (Union[str, Path]): Project directory path.
            readings_dir (Union[str, Path]): Directory path for lesson readings.
            syllabus_path (Union[str, Path]): Path to syllabus document (.pdf or .docx).
            llm (Any): Language model instance for text summarization and relationship extraction.
            course_name (str): Name of the course.
            output_dir (Union[str, Path], optional): Output directory path for generated concept map.
            lesson_range (Union[range, int], optional): Range of lesson numbers to process.
            lesson_objectives (Union[List[str], Dict[str, str]], optional): User-defined lesson objectives.
            verbose (bool, optional): If True, enables verbose logging.
            save_relationships (bool, optional): If True, saves concept relationships as JSON.
            **kwargs: Additional parameters for custom prompts.
        """
        # other setup
        self.llm = llm
        self.course_name = course_name
        self.lesson_range = range(lesson_range, lesson_range + 1) if isinstance(lesson_range, int) else lesson_range
        self.save_relationships = save_relationships
        self.relationship_list = []
        self.concept_list = []
        self.prompts = {'summary': kwargs.get('summary_prompt', summary_prompt),
                        'relationship':  kwargs.get('relationship_prompt', relationship_prompt)}
        self.verbose = verbose
        self.timestamp = datetime.now().strftime("%Y%m%d")

        # set output directory
        rng = [min(self.lesson_range), max(self.lesson_range)]
        if not output_dir:
            self.output_dir = Path(self.lesson_loader.project_dir) / \
                f"ClassFactoryOutput/ConceptWeb/L{rng[0]}_{rng[1]}" if rng[0] != rng[1] else Path(output_dir) / f"L{rng[0]}"
        else:
            self.output_dir = Path(output_dir) / f"L{rng[0]}_{rng[1]}" if rng[0] != rng[1] else Path(output_dir) / f"L{rng[0]}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # load user objectives and readings
        self.user_objectives = self.set_user_objectives(lesson_objectives, self.lesson_range) if lesson_objectives else {}
        self.G = None
        self.readings = self._load_readings(self.lesson_range)

        self.kwargs = kwargs

    def _summarize_document(self, document: str) -> str:
        """
        Summarizes a single document using the LLM.

        Args:
            document (str): Document content to summarize.

        Returns:
            str: Summarized content.
        """
        return summarize_text(document, prompt=self.prompts['summary'], course_name=self.course_name, llm=self.llm)

    def _extract_relationships(self, summary: str, objectives: str) -> List[Tuple[str, str, str]]:
        """
        Extracts relationships between concepts from a summary and objectives.

        Args:
            summary (str): Summarized document content.
            objectives (str): Lesson objectives for context.

        Returns:
            List[Tuple[str, str, str]]: List of relationships as (concept1, relation, concept2) tuples.
        """
        return extract_relationships(summary, objectives, self.course_name, llm=self.llm, verbose=self.verbose)

    def load_and_process_lessons(self, threshold: float = 0.995):
        """
        Process lesson materials by summarizing content and extracting concept relationships.

        Args:
            threshold (float, optional): Similarity threshold for extracted concepts. Defaults to 0.995.

        For each lesson in `lesson_range`:
            - Load documents and objectives.
            - Summarize readings using the LLM.
            - Extract relationships between concepts and generates unique concept list.
        """
        self.logger.info(f"\nLoading lessons from {self.lesson_loader.reading_dir}...")

        # Initialize a new structure to hold readings and summaries
        self.readings_with_summaries = {}

        # summarize readings
        for lesson, readings in self.readings.items():
            lesson_num = int(lesson)
            if not int(lesson_num) in self.lesson_range:
                self.logger.info(f"Lesson {lesson_num} not provided lesson range. Skipping this reading. "
                                 "If this is an error, adjust provided lesson_range")
                continue
            lesson_objectives = self._get_lesson_objectives(lesson_num)

            # Initialize a list to hold summaries for this lesson
            summaries = []

            for document in readings:
                summary = self._summarize_document(document)
                summaries.append(summary)  # Store the summary

                relationships = self._extract_relationships(summary, lesson_objectives)

                self.relationship_list.extend(relationships)
                concepts = extract_concepts_from_relationships(relationships)
                self.concept_list.extend(concepts)

            # Store both readings and summaries in the new structure
            self.readings_with_summaries[lesson] = {
                'readings': readings,
                'summaries': summaries
            }

        # Process relationships to normalize concepts
        self.logger.info("\nExtracting concepts and relations")
        self.relationship_list = process_relationships(self.relationship_list, threshold=threshold)
        self.concept_list = list(set(self.concept_list))  # Ensure unique concepts

    def _save_intermediate_data(self):
        """
        Saves extracted concepts and relationships as JSON files in the output directory. (Triggered if `save_relationships`=True)

        Files saved:
            - `conceptlist_<timestamp>_Lsn_<lesson_range>.json`: List of unique concepts.
            - `relationship_list_<timestamp>_Lsn_<lesson_range>.json`: List of relationships.

        Raises:
            OSError: If saving files fails.
        """
        with open(self.output_dir / f'conceptlist_{self.timestamp}_Lsn_{self.lesson_range}.json', 'w') as f:
            json.dump(self.concept_list, f)

        with open(self.output_dir / f'relationship_list_{self.timestamp}_Lsn_{self.lesson_range}.json', 'w') as f:
            json.dump(self.relationship_list, f)

    def _build_and_visualize_graph(self, method: str = 'leiden', directed: bool = False, dark_mode: bool = True):
        """
        Construct and visualize a concept map graph, including community detection and word cloud generation.

        Steps:
            - Builds a concept graph where nodes represent concepts and edges represent relationships.
            - Detects communities based on the specified `method` ('leiden', 'louvain', or 'spectral').
            - Generates an HTML visualization and word cloud.

        Args:
            method (str, optional): Community detection method. Defaults to 'leiden'.
            directed (bool, optional): If True, creates a directed graph. Defaults to False.
            dark_mode (bool): Sets graph to dark or white background. Defaults to True (dark mode).

        Raises:
            ValueError: If an unrecognized community detection method is used.
        """
        self.logger.info("\nBuilding graph...")
        self.G = build_graph(processed_relationships=self.relationship_list, directed=directed)

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
        visualize_graph_interactive(self.G, output_path=output_html_path, directed=directed, dark_mode=dark_mode)


    def build_concept_map(self, directed: bool = False, concept_similarity_threshold: float = 0.995,
                          dark_mode: bool = True, lesson_objectives: Optional[Dict[str, str]] = None) -> None:
        """
        Execute the full pipeline to generate a concept map.

        Args:
            directed (bool, optional): Whether to create a directed concept map. Defaults to False.
            concept_similarity_threshold (float, optional): Threshold for concept similarity. Defaults to 0.995.
            dark_mode (bool, optional): Whether to use dark mode for visualization. Defaults to True.
            lesson_objectives (Optional[Dict[str, str]], optional): User-provided lesson objectives. Defaults to None.

        Raises:
            ValueError: If any process encounters invalid data.
        """
        self.user_objectives = self.set_user_objectives(lesson_objectives, self.lesson_range) if lesson_objectives else {}
        method = self.kwargs.get('method', 'leiden')
        self.load_and_process_lessons(threshold=concept_similarity_threshold)
        if self.save_relationships:
            self._save_intermediate_data()
        self._build_and_visualize_graph(method=method, directed=directed, dark_mode=dark_mode)


if __name__ == "__main__":
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    # env setup
    from pyprojroot.here import here

    from class_factory.utils.tools import reset_loggers

    reset_loggers()
    load_dotenv()
    user_home = Path.home()

    # Path definitions
    projectDir = here()
    readingDir = user_home / os.getenv('readingsDir')
    syllabus_path = user_home / os.getenv('syllabus_path')
    # pdf_syllabus_path = user_home / os.getenv('pdf_syllabus_path')

    # Example usage
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
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

    loader = LessonLoader(syllabus_path=syllabus_path,
                          reading_dir=readingDir,
                          project_dir=projectDir)

    builder = ConceptMapBuilder(
        lesson_loader=loader,
        llm=llm,
        course_name="American Politics",
        lesson_no=10,
        lesson_range=range(1, 11),
        output_dir=None,
        verbose=False
    )

    builder.build_concept_map(directed=True, dark_mode=False)

# %%
