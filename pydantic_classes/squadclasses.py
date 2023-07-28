from pydantic import BaseModel, Field, validator
from typing import List

class SquadAnswers(BaseModel):
    answer_start: List[int] = Field(description="Answer location in the context property. Location value is 0 based.")
    text: List[str] = Field(description="The answer text from the user input.")
    
    @validator('answer_start')
    def check_answer_start(cls, field):
        if len(field) == 0:
            raise ValueError("No answers starting indexes provided")
        return field
    
    @validator('text')
    def check_text(cls, field):
        if len(field) == 0:
            raise ValueError("No answer text provided")
        return field

class SquadDataItem(BaseModel):
    answers: SquadAnswers = Field(description="Answers for the question.")
    context: str = Field(description="Context used to generate the question and answers from. It comes from the user input provided to you.")
    question: str = Field(description="Question generated from the user input.")
    title: str = Field(description="Title of data set")
    id: str = Field(description="Identifying field for question.")

    @validator('context')
    def check_context(cls, field):
        if len(field) == 0:
            raise ValueError("No context provided")
        return field

    @validator('question')
    def check_question(cls, field):
        if len(field) == 0:
            raise ValueError("No question provided")
        return field  