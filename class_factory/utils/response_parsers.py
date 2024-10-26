from typing import List, Union

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


class Extracted_Relations(BaseModel):
    relationships: List[Relationship] = Field(description="A list of relationships between key concepts")
