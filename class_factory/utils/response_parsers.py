from typing import List, Optional, Union

from pydantic import BaseModel, Field


class MultipleChoiceQuestion(BaseModel):
    question: str = Field(description="The text of the multiple choice question")
    A: str = Field(description="Choice A")
    B: str = Field(description="Choice B")
    C: str = Field(description="Choice C")
    D: str = Field(description="Choice D")
    correct_answer: str = Field(description="The correct answer (A, B, C, or D)")


class TrueFalseQuestion(BaseModel):
    question: str = Field(description="The text of the true/false question")
    A: str = Field(description="Option for 'True'")
    B: str = Field(description="Option for 'False'")
    C: str = Field(default="", description="Blank for C")
    D: str = Field(default="", description="Blank for D")
    correct_answer: str = Field(description="The correct answer (A or B)")


class FillInTheBlankQuestion(BaseModel):
    question: str = Field(description="The text of the fill-in-the-blank question")
    A: str = Field(description="Choice A")
    B: str = Field(description="Choice B")
    C: str = Field(description="Choice C")
    D: str = Field(description="Choice D")
    correct_answer: str = Field(description="The correct answer to fill the blank")


class Quiz(BaseModel):
    multiple_choice: List[MultipleChoiceQuestion] = Field(description="List of multiple choice questions")
    true_false: List[TrueFalseQuestion] = Field(description="List of true/false questions")
    fill_in_the_blank: List[FillInTheBlankQuestion] = Field(description="List of fill-in-the-blank questions")


class Relationship(BaseModel):
    concept_1: str = Field(description="The first concept in the relationship")
    relationship_type: Union[str, None] = Field(description="The type of relationship, or 'None' if no meaningful relationship exists")
    concept_2: str = Field(description="The second concept in the relationship")


class ExtractedRelations(BaseModel):
    relationships: List[Relationship] = Field(description="A list of relationships between key concepts")


# class ValidatorResponse(BaseModel):
#     evaluation_score: float = Field(
#         description="The model's evaluation score, indicating how well the generated content meets the lesson objectives. Scaled from 0 to 10, with higher scores indicating better alignment.")
#     status: int = Field(description="A status code representing the validation outcome. 1 indicates success, while 0 indicates failure or required revisions.")
#     reasoning: str = Field(
#         description="A brief explanation of the validation result, providing feedback on any improvements or issues with the generated content.")
#     additional_guidance: Optional[str] = Field(
#         default=None, description="Optional extra guidance for refining the generated content if revisions are needed.")

# # let's add a field to explainto the model what should be going there


# class ExtractedConcepts(BaseModel):
#     concepts: List[str] = Field(description="A list of concepts extracted from the text")
