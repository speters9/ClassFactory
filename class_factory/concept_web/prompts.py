"""Prompts for llm summarizing and relationship extraction"""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

summary_prompt_system = """
    You are a professor specializing in {course_name}.
    Your role is to analyze academic texts and create structured summaries that will be used to:
    1. Build knowledge graphs connecting key concepts
    2. Generate lecture materials and slides
    3. Create educational resources for undergraduate students

    Focus on capturing the intellectual rigor and theoretical complexity of the source material in a structured, concise, and accessible format.
"""

summary_prompt_human = """
Create a structured summary of the following text, maintaining its academic depth and theoretical nuance.

### Summary Requirements:
- Length: 250-350 words
- Structure your summary with these components:
    1. **Central Thesis**: Primary theoretical argument or scholarly contribution
    2. **Theoretical Framework**: Key theoretical constructs and their relationships
    3. **Supporting Arguments**: Major lines of reasoning and evidence
    4. **Key Findings/Examples**: Empirical evidence or illustrative cases

### Important Considerations:
- Use clear, direct language
- Maintain explicit connections between concepts (e.g., "X leads to Y", "A shapes B")
- Include relevant scholarly context

**Text to Summarize:**
---
{text}
---

{additional_guidance}
"""


relationship_prompt_system = """
    You are a professor specializing in {course_name}.
    Your role is to map relationships between academic concepts and provide structured outputs for undergraduate teaching.
    Your expertise ensures that the extracted concepts and relationships align with lesson objectives and are relevant for an introductory undergraduate {course_name} course.
"""

relationship_prompt_human = """
## You will analyze the text for this lesson to extract *key concepts* and their *relationships* in light of the lesson objectives.

### Lesson Details:
- **Objectives**:
    {objectives}

### Task Instructions:
From the text provided, identify:
1. **Core Theoretical Concepts** (e.g., "State of Nature", "Social Contract")
2. **Causal Mechanisms** (how concepts influence each other)
3. **Structural Relationships** (how concepts fit together)

### Concept Extraction Guidelines:
- Focus on concepts and relationships present in the source text
- Relationships must be directly derived from the source text:
    - Avoid introducing external historical references unless explicitly mentioned in the text.
- Focus on both:
    - Theoretical concepts or important themes (e.g., "Social Contract", "Sovereignty", "Separation of Powers")
    - Causal mechanisms (e.g., "Human Nature", "War", "Competition")
- Include intermediate concepts that connect major ideas
- Ensure concepts form complete theoretical chains
- Each concept should be a significant theoretical idea or mechanism. Examples:
    GOOD: "Separation of Powers"        (fundamental principle)
    BAD:  "House of Representatives"    (too specific: instance of representation)
    GOOD: "Social Contract"             (key theoretical concept)
    BAD:  "Hobbes Leviathan Chapter 2"  (source rather than concept)
    GOOD: "Natural Rights"              (broad theoretical principle)
    BAD:  "Right to Bear Arms"          (specific instance of rights)
- Limit concepts and relationship terms to **no more than three words each** for clarity.

### Relationship Mapping Guidelines:
- Use precise relationship verbs that show:
    - Causation ("leads to", "creates", "produces")
    - Definition ("comprises", "consists of", "characterized by")
    - Support or theoretical connections ("enables", "maintains", "preserves")
- Ensure relationships form complete theoretical pathways

Structure relationships in the format:
```json
"relationships": [
    ["Concept 1", "relationship_type", "Concept 2"],
    ["Concept 1", "relationship_type", "Concept 3"],
    ...
]
```

{additional_guidance}

### IMPORTANT: Your final response must strictly follow this JSON format:
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

### REMINDER:
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
