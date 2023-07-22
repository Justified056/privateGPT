from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from constants import SQUAD_V2_JSON_SCHEMA
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import shutil
import json
import jsonschema
from jsonschema import validate

load_dotenv()

# Load environment variables
source_directory = os.environ.get('CREATE_SQUAD_SOURCE_DIR', 'crawler_text')
squad_data_set_directory = os.environ.get('SQUAD_DATA_SETS_DIR', 'squad_data_sets')
post_processed_directory = os.environ.get('POST_PROCESSED_DATASET_DIR', 'processed_dataset_files')
number_of_files_to_process = int(os.environ.get('NUMBER_OF_FILES_TO_CONVERT_TO_DATASET', 1))
open_api_key = openai_api_key= os.environ.get('PROJECT_OPENAI_API_KEY', '')
print(open_api_key)
files_processed = 0
chunk_size = 500
chunk_overlap = 50
processed_file_path = ""

#This will be the starting name of the file added to the squad_data_set_directory
game_being_processed_file_prefix = "elden_ring"
game_being_processed_file_name = f"{squad_data_set_directory}/{game_being_processed_file_prefix}_walkthrough_as_squad"

def load_squad_data_from_file():
    try:
        with open(game_being_processed_file_name, 'r') as f:
            print(f"Loading existing data from {game_being_processed_file_name}")
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"No data to load file {game_being_processed_file_name} does not exist")
        return [] 

def save_squad_data_to_file (existing_data):
    with open(game_being_processed_file_name, 'w') as f:
        print(f"Saving data to {game_being_processed_file_name}")
        json.dump(existing_data, f)      

def get_document_contents_from_dir() -> str:
    print(f"Loading documents from {source_directory}")
    files = os.listdir(source_directory)
    if not files:  # The directory is empty
        print(f"No documents found in {source_directory}")
        return None

    # Sort files in lexicographical order (alphabetical order)
    files.sort()

    processed_file_path = os.path.join(source_directory, files[0])
    print(f"Processing file: {processed_file_path}")
    with open(processed_file_path, 'r') as file:
        content = file.read()
    
    print(f"Found content length: {len(content)} from {processed_file_path}")
    return content
    
def process_document() -> List[Document]:
    """
    Load document and split in chunks
    """
   
    document_contents = get_document_contents_from_dir()
    if not document_contents:
        print("Exiting due to no document content found")
        exit(1)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(document_contents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

# This operation can fail due to user permissions
def move_document_to_new_folder():
    try:
        if len(processed_file_path) > 0:
            print(f"Moving {processed_file_path} to {post_processed_directory}")
            shutil.move(processed_file_path, post_processed_directory)
        else:
            print(f"No processed file path when calling function to move. Exiting.")
    except IOError as e:
        print(f"Error moving file post processing: IOError: {e}")
        exit(1)
    except shutil.Error as e:
        print(f"Error moving file post processing: ShutilError: {e}")
        exit(1)
    except PermissionError as e:
        print(f"Error moving file post processing: PermissionError: {e}")
        exit(1)

def create_ai_chain(): 
    llm = ChatOpenAI(model="gpt-3.5-turbo", 
                    openai_api_key=open_api_key,
                    temperature=0)

    prompt_msgs = [
        SystemMessage(
            content="You are a intelligent Data Scientist. You are preparing text for training a BERT AI model by creating a SQUAD version 2.0 data set. You will make as many unique question and answers as you can from the input."
        ),
        HumanMessage(
            content="Use the given JSON schema to extract information from the following input:"
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        HumanMessage(content="Tips: Make sure to answer in the JSON schema format provided. Always provide a random GUUID for the id property."),
    ]

    prompt = ChatPromptTemplate(messages=prompt_msgs)
    return create_structured_output_chain(SQUAD_V2_JSON_SCHEMA, llm, prompt) # set verbose=True if you want some debug. Pass it to that function to the left

# Make the chain
chain = create_ai_chain()
#Get existing data from last run
existing_squad_data = load_squad_data_from_file()

while files_processed < number_of_files_to_process:
    try:
        documents = process_document()  
        for document in documents:
            res = chain.run(document) # .run simply returns the output as a string
            print("Validating response from chatGPT returned correct JSON schema.")
            try:
              validate(instance=res, schema=SQUAD_V2_JSON_SCHEMA)
            except json.decoder.JSONDecodeError:
              print('chatGPT returned invalid JSON')
              exit(1)
            except jsonschema.exceptions.ValidationError as ve:
              print('JSON from chatGPT doesn\'t match the schema. Details:', ve)
              exit(1)
            existing_squad_data.append(res)   
        files_processed += 1 
        move_document_to_new_folder()
    except Exception as e:
        print(f"Exception processing file: {processed_file_path} Exception: {e}")
        exit(1)

save_squad_data_to_file(existing_squad_data)