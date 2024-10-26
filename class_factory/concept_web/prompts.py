"""Prompts for llm summarizing and relationship extraction"""

summary_prompt = """
        You are a professor specializing in {course_name}.
        You will be given a text and asked to summarize this text in light of your expertise.
        Your summary will:
            - Focus on the text's **key arguments**, **main points**, and any **significant examples** relevant to an undergraduate {course_name} course.
            - Be between 250-350 words to ensure enough detail is captured, without omitting critical information.
        Summarize the following text: \n {text}
    """

relationship_prompt = """You are a professor specializing in {course_name}.
                        You are instructing an introductory undergraduate {course_name} class.
                        You will be mapping relationships between the concepts this class addresses.
                        The objectives for this lesson are:
                        {objectives}

                        From the following text for this lesson, extract the **key concepts** and the **relationships** between them.
                        \n
                        {text}
                        \n

                        Extract the most important and generally applicable key concepts and themes from the above summary.
                        Focus on high-level concepts or overarching themes relevant to an undergraduate {course_name} course and the lesson objectives.
                        Examples of such concepts might include things like "Separation of Powers", "Federalism", "Standing Armies", or "Representation".

                        Avoid overly specific or narrow topics.

                        Provide the relationships between each concept with the other discovered concepts in the format:
                            "relationships": [
                              ["Concept 1", "relationship_type", "Concept 2"],
                              ["Concept 1", "relationship_type", "Concept 3"],
                              ...
                            ]

                        If there is no meaningful relationship from the standpoint of lesson objectives and your expertise as a professor of {course_name}, \
                        return "None" in the "relationship_type" field.

                        **Limit extracted concepts and relationships to no more than 3 words each**
                        Extract ALL relevant concepts and themes.

                        Because you are comparing each concept to every other concept, the json may be long. That's fine.

                        ### IMPORTANT: Your response **must** strictly follow this JSON format:

                        ```json
                        {{
                          "concepts": [
                            "Concept 1",
                            "Concept 2",
                            "Concept 3",
                            ...
                          ],
                          "relationships": [
                            ["Concept 1", "relationship_to_Concept_2", "Concept 2"],
                            ["Concept 1", "relationship_to_Concept_3", "Concept 3"],
                            ["Concept 2", "relationship_to_Concept_3", "Concept 3"],
                            ...
                          ]
                        }}
                        ```

                        ### IMPORTANT: Your response **must** strictly follow the JSON format above. Include only the json in your response.
                        If the JSON is invalid or extra text is included, your response will be rejected and you will not be paid.
                        """

no_objective_relationship_prompt = """You are a political science professor specializing in {course_name}.
                        You are instructing an introductory undergraduate {course_name} class.
                        You will be mapping relationships between the concepts this class addresses.

                        From the following text for this lesson, extract the **key concepts** and the **relationships** between them.
                        \n
                        {text}
                        \n

                        Extract the most important and generally applicable key concepts and themes from the above summary.
                        Focus on high-level concepts or overarching themes relevant to an undergraduate {course_name} course and the lesson objectives.
                        Examples of such concepts might include things like "Separation of Powers", "Federalism", "Standing Armies", or "Representation".

                        Avoid overly specific or narrow topics.

                        Provide the relationships between each concept with the other discovered concepts in the format:
                            "relationships": [
                              ["Concept 1", "relationship_type", "Concept 2"],
                              ["Concept 1", "relationship_type", "Concept 3"],
                              ...
                            ]

                        If there is no meaningful relationship from the standpoint of lesson objectives and your expertise as a professor of {course_name}, \
                        return "None" in the "relationship_type" field.

                        **Limit extracted concepts and relationships to no more than 3 words each**
                        Extract ALL relevant concepts and themes.

                        Because you are comparing each concept to every other concept, the json may be long. That's fine.

                        ### IMPORTANT: Your response **must** strictly follow this JSON format:

                        ```json
                        {{
                          "concepts": [
                            "Concept 1",
                            "Concept 2",
                            "Concept 3",
                            ...
                          ],
                          "relationships": [
                            ["Concept 1", "relationship_to_Concept_2", "Concept 2"],
                            ["Concept 1", "relationship_to_Concept_3", "Concept 3"],
                            ["Concept 2", "relationship_to_Concept_3", "Concept 3"],
                            ...
                          ]
                        }}
                        ```

                        ### IMPORTANT: Your response **must** strictly follow the JSON format above. Include only the json in your response.
                        If the JSON is invalid or extra text is included, your response will be rejected and you will not be paid.
                        """
