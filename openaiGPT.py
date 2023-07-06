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
from prompts import get_chain

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    index = VectorStoreIndexWrapper(vectorstore=db)
    chain = get_chain(index.vectorstore) 

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