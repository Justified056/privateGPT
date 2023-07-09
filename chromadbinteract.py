import chromadb
from constants import CHROMA_SETTINGS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import pprint

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")

def main():
  client = chromadb.Client(settings=CHROMA_SETTINGS)
  embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
  collection = client.get_collection("langchain", embedding_function=embeddings.embed_documents)
  queryResults = collection.query(query_texts=["What attributes does the Warrior class start with?"],
                                  n_results=10
                                )
  pprint.pprint(queryResults)
if __name__ == "__main__":
    main()