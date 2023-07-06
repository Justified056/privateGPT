from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question is about a video game.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about video games.
You are given the following extracted parts of long documents and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about a video game, politely inform them that you are tuned to only answer questions about video games.
Question: {question}
=========
Context: {context}
=========
Answer:
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    load_dotenv()
    llm = ChatOpenAI(model="gpt-3.5-turbo", 
                     openai_api_key= os.environ.get('OPENAI_API_KEY'),
                     temperature=0)
    
    doc_chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=QA_PROMPT
    )
    question_chain = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
    )
    return ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_chain,
        return_source_documents=True
    )