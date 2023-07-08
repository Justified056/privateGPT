import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
import argparse
import time
from constants import CHROMA_SETTINGS
from prompts import get_chain
from datetime import datetime

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

def save_chat_history(chat_history):
    # Get the current date
    current_date = datetime.now().strftime("%Y%m%d")
    
    # Define the file name
    file_name = f"chat_history_logs/openaigpt_{current_date}.txt"
    
    # Check if the file already exists
    if os.path.exists(file_name):
        # File exists, append to it
        mode = "a"
    else:
        # File doesn't exist, create a new one
        mode = "w"
    
    # Open the file in the appropriate mode
    with open(file_name, mode) as file:
        for query, answer in chat_history:
            # Write the query and answer to the file
            file.write(f"Query: {query}\nAnswer: {answer}\n")
    
    print(f"Chat history saved to {file_name}")

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    index = VectorStoreIndexWrapper(vectorstore=db)
    # similarity search kwordargs search_kwargs = {'k': 10}
    # similarity score threshold search_type="similarity_score_threshold", search_kwargs={"score_threshold": .7, "k": 10}
    chain = get_chain(index.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})) 

    # Interactive questions and answers
    chat_history = []
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            save_chat_history(chat_history)
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = chain({"question": query, "chat_history": chat_history})
        answer, docs = res['answer'], [] if args.hide_source else res['source_documents']
        end = time.time()
        chat_history.append((query, answer))
        # Print the relevant sources used for the answer
        for document in docs:
            metaDataValuesDisplay = ""
            for key, value in document.metadata.items():
                metaDataValuesDisplay += f'{key}: {value}, '
            print("\n> " + f'Embedding metadata: {metaDataValuesDisplay.rstrip(", ")}')
            print(f'Content: {document.page_content}')

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

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