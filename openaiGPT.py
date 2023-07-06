import os
import sys
from dotenv import load_dotenv
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import argparse
import time
from constants import CHROMA_SETTINGS

load_dotenv()

tempalte = """I want you to ANSWER a QUESTION based on the following pieces of CONTEXT

              If you don't know the answer, just say that you don't know, don't try to make up an answer.

              Your ANSWER should be truthful and correct according to the given SOURCES.

              CONTEXT: {context}
              
              Question: {question} 
              
              ANSWER:
              """

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
openai_key = os.environ.get('OPENAI_API_KEY')

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    index = VectorStoreIndexWrapper(vectorstore=db)
    retriever = index.vectorstore.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
        
    chain = ConversationalRetrievalChain.from_llm(
      llm=ChatOpenAI(model="gpt-3.5-turbo", 
                    openai_api_key=openai_key),
                    retriever=retriever,
                    callbacks=callbacks)
    
    # Interactive questions and answers
    chat_history = []
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = chain({"question": query, "chat_history": chat_history})
        #answer, docs = res['answer'], [] if args.hide_source else res['source_documents']
        answer = res['answer']
        end = time.time()
        chat_history.append((query, answer))
        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        """for document in docs:
            metaDataValuesDisplay = ""
            for key, value in document.metadata.items():
                metaDataValuesDisplay += f'{key}: {value}, '
            print("\n> " + f'Embedding metadata: {metaDataValuesDisplay.rstrip(", ")}')
            print(f'Content: {document.page_content}')"""

def parse_arguments():
    parser = argparse.ArgumentParser(description='openAi: Ask questions to your documents online, '
                                                 'using the power of OpenAis ChatGPT.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()