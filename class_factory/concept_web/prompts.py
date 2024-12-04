"""Prompts for llm summarizing and relationship extraction"""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

# summary_prompt = """
#         You are a professor specializing in {course_name}.
#         You will be given a text and asked to summarize this text in light of your expertise.
#         Your summary will:
#             - Focus on the text's **key arguments**, **main points**, and any **significant examples** relevant to an undergraduate {course_name} course.
#             - Be between 250-350 words to ensure enough detail is captured, without omitting critical information.
#         Summarize the following text: \n {text}
#     """

summary_prompt_system = """
    You are a professor specializing in {course_name}.
    Your role is to analyze academic texts and generate summaries that focus on key arguments, main points, and significant examples.
    These summaries are designed for an undergraduate {course_name} course.
"""

summary_prompt_human = """
    Your task is to summarize the following text for an undergraduate {course_name} class.

    ### Summary Guidelines:
    - Focus on the text's **key arguments**, **main points**, and any **significant examples**.
    - Keep the summary between **250-350 words** to ensure sufficient detail while avoiding unnecessary elaboration.

        **Text to Summarize:**
        ---
        {text}
        ---
"""


relationship_prompt_system = """
    You are a professor specializing in {course_name}.
    Your role is to map relationships between academic concepts and provide structured outputs for undergraduate teaching.
    Your expertise ensures that the extracted concepts and relationships align with lesson objectives and are relevant for an introductory undergraduate {course_name} course.
"""

relationship_prompt_human = """
    You will analyze the text for this lesson to extract **key concepts** and their **relationships** in light of the lesson objectives.

    ### Lesson Details:
    - **Objectives**:
      {objectives}

    ### Task Instructions:
        From the text provided, identify:
        1. The **concepts** that pertain to the lesson objectives.
        2. The **relationships** between these concepts.

    ### Concept Extraction Guidelines:
        - Focus on **high-level** or **overarching concepts** (e.g., "Separation of Powers", "Representation", or "Federalism").
        - Avoid overly specific, narrow, or redundant topics.
        - **There is no upper limit** on the number of concepts you may return, provided that:
          - Each concept meets the criteria of being **high-level**.
          - The list of concepts is relevant to the lesson objectives and the text.
        - Limit concepts and relationship terms to **three words each** for clarity.

    ### Relationship Mapping Guidelines:
        - Structure relationships in the format:
          ```json
          "relationships": [
            ["Concept 1", "relationship_type", "Concept 2"],
            ["Concept 1", "relationship_type", "Concept 3"],
            ...
          ]

    {additional_guidance}

    ### IMPORTANT: Your final response **must** strictly follow this JSON format:

        ```json
        {{
          "concepts": [
            "Concept 1",
            "Concept 2",
            "Concept 3",
            "Concept 4",
            ...
          ],
          "relationships": [
            ["Concept 1", "relationship_to_Concept_2", "Concept 2"],
            ["Concept 1", "relationship_to_Concept_3", "Concept 3"],
            ["Concept 1", "relationship_to_Concept_4", "Concept 4"],
            ["Concept 2", "relationship_to_Concept_3", "Concept 3"],
            ["Concept 2", "relationship_to_Concept_4", "Concept 4"],
            ...
          ]
        }}
        ```

    ### IMPORTANT:
        - Ensure your final response strictly adheres to the JSON format provided. Include only the json in your response.
        - If the JSON is invalid or extra text is included, your response will be rejected.
"""


no_objective_relationship_prompt_human = """
    You will analyze the text for this lesson to extract **key concepts** and their **relationships**.

    ### Lesson Details:
    - **Objectives**:
      None Provided

    ### Task Instructions:
        From the text provided, identify:
        1. The **concepts** found within the text.
        2. The **relationships** between these concepts.

    ### Concept Extraction Guidelines:
        - Focus on high-level or overarching concepts (e.g., "Separation of Powers", "Representation", or "Federalism").
        - Avoid overly specific or narrow topics.
        - Limit concepts and relationship terms to **three words each**.

    ### Relationship Mapping Guidelines:
        - Structure relationships in the format:
          ```json
          "relationships": [
            ["Concept 1", "relationship_type", "Concept 2"],
            ["Concept 1", "relationship_type", "Concept 3"],
            ...
          ]

    {additional_guidance}

    ### IMPORTANT: Your final response **must** strictly follow this JSON format:

        ```json
        {{
          "concepts": [
            "Concept 1",
            "Concept 2",
            "Concept 3",
            "Concept 4",
            ...
          ],
          "relationships": [
            ["Concept 1", "relationship_to_Concept_2", "Concept 2"],
            ["Concept 1", "relationship_to_Concept_3", "Concept 3"],
            ["Concept 1", "relationship_to_Concept_4", "Concept 4"],
            ["Concept 2", "relationship_to_Concept_3", "Concept 3"],
            ["Concept 2", "relationship_to_Concept_4", "Concept 4"],
            ...
          ]
        }}
        ```

    ### IMPORTANT:
        - Ensure your final response strictly adheres to the JSON format provided. Include only the json in your response.
        - If the JSON is invalid or extra text is included, your response will be rejected.
"""


# Summary Prompt
summary_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(summary_prompt_system),
        HumanMessagePromptTemplate.from_template(summary_prompt_human)
    ]
)


relationship_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(relationship_prompt_system),
        HumanMessagePromptTemplate.from_template(relationship_prompt_human)
    ]
)


no_objective_relationship_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(relationship_prompt_system),
        HumanMessagePromptTemplate.from_template(no_objective_relationship_prompt_human)
    ]
)

# relationship_prompt = """You are a professor specializing in {course_name}.
#                         You are instructing an introductory undergraduate {course_name} class.
#                         You will be mapping relationships between the concepts this class addresses.
#                         The objectives for this lesson are:
#                         {objectives}

#                         From the following text for this lesson, extract the **key concepts** and the **relationships** between them.
#                         \n
#                         {text}
#                         \n

#                         Extract the most important and generally applicable key concepts and themes from the above summary.
#                         Focus on high-level concepts or overarching themes relevant to an undergraduate {course_name} course and the lesson objectives.
#                         Examples of such concepts might include things like "Separation of Powers", "Federalism", "Standing Armies", or "Representation".

#                         Avoid overly specific or narrow topics.

#                         Provide the relationships between each concept with each other discovered concept in the format:
#                             "relationships": [
#                               ["Concept 1", "relationship_type", "Concept 2"],
#                               ["Concept 1", "relationship_type", "Concept 3"],
#                               ...
#                             ]

#                         Use specific relational terms for concepts. Avoid broad relationship descriptors.

#                         If there is no meaningful relationship from the standpoint of lesson objectives and your expertise as a professor of {course_name}, \
#                         return "None" in the "relationship_type" field.

#                         **Limit extracted concepts and relationships to no more than 3 words each**
#                         Extract ALL relevant concepts and themes.

#                         Because you are comparing each concept to every other concept, the json may be long. That's fine.

#                         {additional_guidance}

#                         ### IMPORTANT: Your response **must** strictly follow this JSON format:

#                         ```json
#                         {{
#                           "concepts": [
#                             "Concept 1",
#                             "Concept 2",
#                             "Concept 3",
#                             "Concept 4",
#                             ...
#                           ],
#                           "relationships": [
#                             ["Concept 1", "relationship_to_Concept_2", "Concept 2"],
#                             ["Concept 1", "relationship_to_Concept_3", "Concept 3"],
#                             ["Concept 1", "relationship_to_Concept_4", "Concept 4"],
#                             ["Concept 2", "relationship_to_Concept_3", "Concept 3"],
#                             ["Concept 2", "relationship_to_Concept_4", "Concept 4"],
#                             ...
#                           ]
#                         }}
#                         ```

#                         ### IMPORTANT: Your response **must** strictly follow the JSON format above. Include only the json in your response.
#                         If the JSON is invalid or extra text is included, your response will be rejected.
#                         """

# no_objective_relationship_prompt = """You are a political science professor specializing in {course_name}.
#                         You are instructing an introductory undergraduate {course_name} class.
#                         You will be mapping relationships between the concepts this class addresses.

#                         From the following text for this lesson, extract the **key concepts** and the **relationships** between them.
#                         \n
#                         {text}
#                         \n

#                         Extract the most important and generally applicable key concepts and themes from the above summary.
#                         Focus on high-level concepts or overarching themes relevant to an undergraduate {course_name} course and the lesson objectives.
#                         Examples of such concepts might include things like "Separation of Powers", "Federalism", "Standing Armies", or "Representation".

#                         Avoid overly specific or narrow topics.

#                         Provide the relationships between each concept with the other discovered concepts in the format:
#                             "relationships": [
#                               ["Concept 1", "relationship_type", "Concept 2"],
#                               ["Concept 1", "relationship_type", "Concept 3"],
#                               ...
#                             ]

#                         Use specific relational terms for concepts. Avoid broad relationship descriptors.

#                         If there is no meaningful relationship from the standpoint of lesson objectives and your expertise as a professor of {course_name}, \
#                         return "None" in the "relationship_type" field.

#                         **Limit extracted concepts and relationships to no more than 3 words each**
#                         Extract ALL relevant concepts and themes.

#                         Because you are comparing each concept to every other concept, the json may be long. That's fine.

#                         {additional_guidance}

#                         ### IMPORTANT: Your response **must** strictly follow this JSON format:

#                         ```json
#                         {{
#                           "concepts": [
#                             "Concept 1",
#                             "Concept 2",
#                             "Concept 3",
#                             "Concept 4",
#                             ...
#                           ],
#                           "relationships": [
#                             ["Concept 1", "relationship_to_Concept_2", "Concept 2"],
#                             ["Concept 1", "relationship_to_Concept_3", "Concept 3"],
#                             ["Concept 1", "relationship_to_Concept_4", "Concept 4"],
#                             ["Concept 2", "relationship_to_Concept_3", "Concept 3"],
#                             ["Concept 2", "relationship_to_Concept_4", "Concept 4"],
#                             ...
#                           ]
#                         }}
#                         ```

#                         ### IMPORTANT: Your response **must** strictly follow the JSON format above. Include only the json in your response.
#                         If the JSON is invalid or extra text is included, your response will be rejected.
#                         """
