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
</instructions>

<text_to_summarize>
{text}
</text_to_summarize>

<summary_requirements>
Length: 250-350 words
Structure your summary with these components:
    1. **Central Thesis**: Primary theoretical argument or scholarly contribution
    2. **Theoretical Framework**: Key theoretical constructs and their relationships
    3. **Supporting Arguments**: Major lines of reasoning and evidence
    4. **Key Findings/Examples**: Empirical evidence or illustrative cases
</summary_requirements>


<important_considerations>
Use clear, direct language
Maintain explicit connections between concepts (e.g., "X leads to Y", "A shapes B")
Include relevant scholarly context
</important_considerations>

<additional_guidance>
{additional_guidance}
</additional_guidance>
"""


relationship_prompt_system = """
    You are a professor specializing in {course_name} who is building a concept map (knowledge graph of concepts) for their students.
    Your role is to map relationships between academic concepts and provide structured outputs that will be used to build the concept map.
    Be strict about using concepts and relationships present in the text.
"""

relationship_prompt_human = """
<instructions>
You are presented with a detailed summary of a particular course reading. The summary has been structured to highlight key concepts and their interrelations.
Analyze the text for this lesson to extract *key theoretical concepts and their interrelationships* in light of the lesson objectives.
</instructions>

<text_to_analyze>
{text}
</text_to_analyze>

<lesson_objectives>
{objectives}
</lesson_objectives>


<task_instructions>
From the text provided, identify:
- **Key theoretical concepts** and the **relationships** between them, focusing on how concepts fit together and influence each other.
</task_instructions>

<relationship_extraction_guidelines>
- Focus only on concepts and relationships present in the source text.
- Relationships must be directly derived from the source text:
    - Avoid introducing external historical references unless explicitly mentioned in the text.

- In your relationship extraction, focus on both:
    - Theoretical concepts or important themes (e.g., "Social Contract", "Sovereignty", "Separation of Powers")
    - Causal mechanisms (e.g., "Human Nature", "War", "Competition")

- Ensure concepts and their relations form complete theoretical chains (e.g. you can proxy this by ensuring the extracted relationship forms a sentence)
- Each concept should be a significant theoretical idea or mechanism. Examples:
    GOOD: "Separation of Powers"        (fundamental principle)
    BAD:  "House of Representatives"    (too specific: instance of representation)
    GOOD: "Social Contract"             (key theoretical concept)
    BAD:  "Hobbes Leviathan Chapter 2"  (source rather than concept)
    GOOD: "Natural Rights"              (broad theoretical principle)
    BAD:  "Right to Bear Arms"          (specific instance of rights)

- Normalize concept names to Title Case; keep to 1-3 words; remove filler words ("of","the","and") unless part of a named theory.
- Do not introduce proper names (people, countries, years) as concepts unless the text frames them as general theories or ideologies.
- Treat <lesson_objectives> (if included) as prioritization hints only; never invent relationships that are found only in the objectives.
- Singularize concept names unless the plural is the named term (e.g., 'Civil-Military Relations').
- Merge aliases to one canonical form (e.g., 'Objective Civilian Control' → 'Objective Control' or 'Professional Military Ethic' → 'Professional Ethic').
- Keep hyphenated named terms (e.g., 'Civil-Military Relations') as written.
</relationship_extraction_guidelines>

<relationship_mapping_guidelines>
- The extracted relationships should take the form ['concept_1', 'relationship', 'concept_2']
- Use precise relationship verbs that show:
    - Causation
    - Definition
    - Support or theoretical connections
- Here is a set of examples that might be used as a relationship_type:
  ["causes","leads to","increases","reduces","requires","enables","preserves",
   "undermines","supports","depends on","comprises","includes","contrasts with",
   "aligns with","conflicts with","achieved by"].
- Make sure to draw as many connections as possible between various concepts, but do not invent content that is not supported by the text.
- If you find concepts that are not connected to the main theoretical chain, look for plausible, text-supported relationships or intermediate concepts that could bridge them. Do not invent content, but do make explicit any connections that are implied or can be reasonably inferred from the text.
- Include intermediate concepts that connect major ideas and help reduce isolated or disconnected concepts in the map.
- Return at most 20 triples; choose the most central/connected relations when eliminating extra extracted relations.
</relationship_mapping_guidelines>

<additional_guidance>
{additional_guidance}
</additional_guidance>

<example_summary_and_relationships>

<example_summary>
**Structured Summary of Huntington (1957), Chapter 4: Power, Professionalism, and Ideology**\n\n1. **Central Thesis** \nHuntington advances a nuanced theoretical distinction in civil-military relations between *subjective* and *objective* civilian control. He argues that genuine civilian control is best conceptualized as the maximization of military professionalism (objective control), rather than the mere maximization of civilian power over the military (subjective control). This reframing resolves longstanding ambiguities in the concept of civilian control and clarifies the conditions under which military professionalism and effective civilian oversight coexist.\n\n2. **Theoretical Framework** \nThe framework hinges on three interrelated constructs: \n- **Civilian Control**: Divided into *subjective* (civilian groups maximize their own power over the military, often at the expense of other civilian groups) and *objective* (civilian control is achieved by fostering a politically neutral, professional military). \n- **Military Professionalism**: Defined as the military’s autonomous functional expertise and political neutrality, which minimizes military political power while preserving effective military capacity. \n- **Political Ideology and Power Relations**: The compatibility between the military professional ethic and prevailing civilian ideologies (liberalism, fascism, Marxism, conservatism) shapes the equilibrium of civil-military relations. Power is analyzed in terms of formal authority (position within government hierarchy) and informal influence (social affiliations, resources, prestige).\n\n3. **Supporting Arguments** \n- *Subjective civilian control* is historically prevalent but inherently unstable and politically partial, as it reflects struggles among civilian factions rather than a principled control over the military. It often undermines military effectiveness and security. \n- *Objective civilian control* requires recognizing the military as a distinct professional group with its own functional imperatives, thereby depoliticizing the military and reducing its political power to a minimum consistent with professionalism. \n- The military’s political power and professionalism are mediated by the ideological environment: antimilitary ideologies (liberalism, fascism, Marxism) tend to force the military into political roles that dilute professionalism, while conservative ideologies are more compatible with professional military values. \n- Power relations are multidimensional, involving the level (hierarchical position), unity (structural cohesion), and scope (range of authority) of military authority, as well as informal influence through social ties and resources.\n\n4. **Key Findings/Examples** \n- Historical examples illustrate subjective control: parliamentary struggles over military power in 17th-18th century England, aristocratic vs. bourgeois control in Europe, and competing claims of U.S. Congress and Presidency. \n- Objective control emerged with the rise of the military profession, exemplified by Prussia/Germany’s Bismarckian era, where high military professionalism coexisted with significant military political power under a pro-military ideology. \n- Antimilitary ideology combined with high military political power and low professionalism characterizes unstable or developing states (e.g., Japan pre-WWII, U.S. during WWII). \n- The U.S. from post-Civil War to WWII exemplified antimilitary ideology with low military power but high professionalism, reflecting objective civilian control. \n- The military ethic contrasts sharply with liberalism, fascism, and Marxism in views on human nature, power, and war, but aligns closely with conservatism, which shares its realism and acceptance of power and institutional continuity. \n- The equilibrium of civil-military relations is dynamic and contingent on ideological compatibility and power distribution; military professionalism is undermined when the military becomes politically involved or when civilian groups impose their interests subjectively.\n\n---\n\nThis summary captures Huntington’s rigorous conceptual distinctions and their implications for understanding the complex interplay of power, ideology, and professionalism in civil-military relations. It highlights the theoretical innovation of objective civilian control as a politically neutral standard and situates it within broader ideological and historical contexts.
</example_summary>

<relationships_extracted_from_example_summary>
{{
  "relationships": [
    {{"concept_1": "Civilian Control", "relationship_type": "comprises", "concept_2": "Objective Control"}},
    {{"concept_1": "Civilian Control", "relationship_type": "comprises", "concept_2": "Subjective Control"}},
    {{"concept_1": "Objective Control", "relationship_type": "requires", "concept_2": "Military Professionalism"}},
    {{"concept_1": "Objective Control", "relationship_type": "reduces", "concept_2": "Military Politicization"}},
    {{"concept_1": "Objective Control", "relationship_type": "preserves", "concept_2": "Military Capacity"}},
    {{"concept_1": "Objective Control", "relationship_type": "contrasts with", "concept_2": "Subjective Control"}},
    {{"concept_1": "Objective Control", "relationship_type": "achieved by", "concept_2": "Depoliticization"}},
    {{"concept_1": "Military Professionalism", "relationship_type": "requires", "concept_2": "Political Neutrality"}},
    {{"concept_1": "Military Professionalism", "relationship_type": "requires", "concept_2": "Functional Expertise"}},
    {{"concept_1": "Military Professionalism", "relationship_type": "minimizes", "concept_2": "Military Political Power"}},
    {{"concept_1": "Military Professionalism", "relationship_type": "enables", "concept_2": "Effective Civilian Control"}},
    {{"concept_1": "Subjective Control", "relationship_type": "maximizes", "concept_2": "Civilian Faction Power"}},
    {{"concept_1": "Subjective Control", "relationship_type": "undermines", "concept_2": "Military Effectiveness"}},
    {{"concept_1": "Political Ideology", "relationship_type": "shapes", "concept_2": "Military Professionalism"}},
    {{"concept_1": "Political Ideology", "relationship_type": "shapes", "concept_2": "Military Political Power"}},
    {{"concept_1": "Antimilitary Ideology", "relationship_type": "increases", "concept_2": "Military Politicization"}},
    {{"concept_1": "Antimilitary Ideology", "relationship_type": "dilutes", "concept_2": "Military Professionalism"}},
    {{"concept_1": "Civil-Military Equilibrium", "relationship_type": "depends on", "concept_2": "Ideological Compatibility"}},
    {{"concept_1": "Civil-Military Equilibrium", "relationship_type": "depends on", "concept_2": "Power Distribution"}},
    {{"concept_1": "Military Ethic", "relationship_type": "aligns with", "concept_2": "Conservative Ideology"}}
  ]
}}
</relationships_extracted_from_example_summary>

</example_summary_and_relationships>


<important>
- Think before responding.
- Use consistent naming conventions for concepts and relationships.
- Do not include author names as concepts. Instead, extract and name the core theoretical ideas, frameworks, or models associated with those authors.
- Named theories (e.g., "Principal-Agent Theory", "Objective Control Theory") are acceptable as concepts, but do not use just the author's name (e.g., "Huntington") as a concept.
- Never return example or placeholder relationships. Only extract real relationships present in the provided text.
- Your final response must include a top-level `relationships` array matching the required schema. Do not include extra text or any other top-level fields.
</important>

Now, extract relationships from the provided text.
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
