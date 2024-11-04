import logging
from typing import Any, Dict

from langchain_core.prompts import PromptTemplate

from class_factory.utils.tools import logger_setup, retry_on_json_decode_error


class Validator:
    """A class for validating responses generated by an LLM (Language Model).

    The Validator checks the accuracy, completeness, and relevance of the LLM's response
    to ensure it meets the requirements specified in the task prompt. Validation results
    include a score, status, reasoning, and any additional guidance.
    """

    def __init__(self, llm: Any, parser: Any, temperature: float = 0.6) -> None:
        """
        Initialize the Validator instance.

        Args:
            llm (Any): The language model instance used to generate responses.
            parser (Any): The parser instance to process the LLM's output. Common parsers might include JsonOutputParser with associated pydantic object.
            temperature (float, optional): The temperature setting for response variability. Defaults to 0.6.
        """
        self.llm = llm
        self.parser = parser
        self.llm.temperature = temperature
        self.logger = logger_setup(logger_name="validator", log_level=logging.INFO)

        # Send the prompt to the LLM

    @retry_on_json_decode_error()
    def validate(self, task_description: str, generated_response: str, specific_guidance: str = "") -> Dict[str, Any]:
        """
        Validates a generated response by providing the task description, the generated response, and any specific guidance for evaluation.

        Args:
            task_description (str): Description of the task that the LLM was originally given.
            generated_response (str): The output generated by the LLM that needs validation.
            specific_guidance (str, optional): Additional guidance for the LLM during validation. Defaults to "".

        Returns:
            Dict[str, Any]: Validation result with keys such as "evaluation_score", "status", "reasoning",
            and "additional_guidance", providing feedback on the response's quality and fit for the task.
        """
        prompt_template = """
            You are given the prompt of an AI Agent and its generated response.
            Your task is to evaluate whether the response is accurate, complete, and fulfills the requirements of the task.
            Focus on evaluating the content of the response, not its format.

            {specific_guidance}

            Original task prompt:
            "{task_description}"

            Generated response:
            "{generated_response}"


            Your evaluation should include:
            - "evaluation_score" - a score from 0.0 to 10, where 10 is the best score and 0 is the worst.
            - "status" - 1 if the response fits the task requirements, 0 otherwise (score must be greater than 7 to be valid).
            - "reasoning" - a brief explanation of how you determined the evaluation_score.
            - "additional_guidance" - if "status" is 0, suggest specific updates to the task description to help the AI Agent generate a more accurate response.

            Example output format if status is 1 (JSON):
                {{
                "evaluation_score": 8.5,
                "status": 1,
                "reasoning": "The response adequately summarizes the main points from the original text and aligns with the task.",
                "additional_guidance": ""
                }}

            Example output format if status is 0 (JSON):
                {{
                "evaluation_score": 6.5,
                "status": 0,
                "reasoning": "The response does not adequately summarize the main points from the original text. It misses important concepts.",
                "additional_guidance": "Ensure you discuss all of the key concepts in the text."
                }}

            Respond only with valid JSON format containing "evaluation_score", "status", "reasoning", and "additional_guidance".

            ### IMPORTANT: Your response **must** strictly follow the JSON format above. Include only the json in your response.
            If the JSON is invalid or extra text is included, your response will be rejected.
            """

        prompt = PromptTemplate.from_template(prompt_template)

        # Format the prompt with the provided data
        chain = prompt | self.llm | self.parser

        # Send the prompt to the LLM
        response = chain.invoke({
            "task_description": task_description,
            "generated_response": generated_response,
            "specific_guidance": specific_guidance
        })

        return response


if __name__ == "__main__":
    import json
    import os
    from pathlib import Path

    # env setup
    from dotenv import load_dotenv
    # llm chain setup
    from langchain_community.llms import Ollama
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_openai import ChatOpenAI
    from pyprojroot.here import here

    from class_factory.concept_web.concept_extraction import (
        extract_relationships, summarize_text)
    from class_factory.concept_web.prompts import (relationship_prompt,
                                                   summary_prompt)
    from class_factory.utils.load_documents import (extract_lesson_objectives,
                                                    load_lessons)
    from class_factory.utils.response_parsers import (Extracted_Relations,
                                                      ValidatorResponse)
    projectDir = here()
    load_dotenv()

    # Path definitions
    user_home = Path.home()
    readingDir = user_home / os.getenv('readingsDir')
    syllabus_path = user_home / os.getenv('syllabus_path')

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
    #     temperature=0.2
    # )

    parser = JsonOutputParser(pydantic_object=Extracted_Relations)
    val_parser = JsonOutputParser(pydantic_object=ValidatorResponse)
    course_name = "American Government"

    validator = Validator(llm=llm, parser=val_parser)
    # Load documents and lesson objectives
    for lesson_num in range(19, 20):
        lesson_objectives = extract_lesson_objectives(syllabus_path, lesson_num, only_current=True)
        documents = load_lessons(readingDir, lesson_range=range(lesson_num, lesson_num + 1), recursive=True)

        for document in documents:
            retries = 0
            additional_guidance = ""
            valid = False
            summary = summarize_text(document, prompt=summary_prompt, course_name=course_name, llm=llm)

            combined_template = PromptTemplate.from_template(relationship_prompt)
            chain = combined_template | llm | parser

            while not valid and retries < 3:
                response = chain.invoke({'course_name': course_name,
                                         'objectives': lesson_objectives,
                                         'text': document,
                                         'additional_guidance': additional_guidance})

                # Clean and parse the JSON output
                if isinstance(response, str):
                    response_cleaned = response.replace("```json", "").replace("```", "")
                    data = json.loads(response_cleaned)  # This may raise JSONDecodeError
                else:
                    data = response

                # Verify that data is a dict
                if not isinstance(data, dict):
                    raise ValueError("Parsed data is not a dictionary.")

                response_str = json.dumps(response).replace("{", "{{").replace("}", "}}")

                validation_prompt = combined_template.format(course_name=course_name,
                                                             objectives=lesson_objectives,
                                                             text=document,
                                                             additional_guidance=additional_guidance).replace("{", "{{").replace("}", "}}")

                val_response = validator.validate(task_description=validation_prompt,
                                                  generated_response=response_str)

                print(f"validation output: {val_response}")
                if int(val_response['status']) == 1:
                    valid = True
                else:
                    retries += 1
                    additional_guidance = val_response.get("additional_guidance", "")
                    validator.logger.warning("Validation failed, attempting retry")

            if valid:
                validator.logger.info("Validation succeeded.")
            else:
                raise ValueError("Validation failed after max retries. Ensure correct prompt and input data. Consider use of a different LLM.")
