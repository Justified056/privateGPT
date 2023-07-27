from pydantic import BaseModel, Field, validator
from typing import List

class SquadAnswers(BaseModel):
    answer_start: List[int] = Field(description="Answer location in the user input. The value is the starting index of the text property. Index values are 0 based indexes. Must be populated")
    text: List[str] = Field(description="The answer text from the user input. Must be populated.")
    
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
    answers: SquadAnswers = Field(description="Answers for the question. Must be populated")
    context: str = Field(description="Context used to generate the question and answers from. It comes from the user input provided to you. Must be populated.")
    question: str = Field(description="Question generated from the user input. Must be populated.")
    title: str = Field(description="Do not set this property. The user will provide a value.")
    id: str = Field(description="Do not set this property. The user will provide a value.")

    def strip(value:any):
        """Do nothing...Langchaing keeps calling this for some reason.""" 

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