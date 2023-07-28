from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic_classes.squadclasses import SquadDataItem
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser, RetryWithErrorOutputParser
from dotenv import load_dotenv
from constants import SQUAD_V2_JSON_SCHEMA
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import pickle
import json
import jsonschema
import uuid
import traceback

load_dotenv()

# Load environment variables
source_directory = os.environ.get('CREATE_SQUAD_SOURCE_DIR', 'crawler_text')
squad_data_set_directory = os.environ.get('SQUAD_DATA_SETS_DIR', 'squad_data_sets')
post_processed_directory = os.environ.get('POST_PROCESSED_DATASET_DIR', 'processed_dataset_files')
number_of_files_to_process = int(os.environ.get('NUMBER_OF_FILES_TO_CONVERT_TO_DATASET', 1))
open_api_key = openai_api_key= os.environ.get('PROJECT_OPENAI_API_KEY', '')
files_processed = 0
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
print(f"Loading documents from {source_directory}")
SOURCE_FILES = os.listdir(source_directory)
# Sort files in lexicographical order (alphabetical order)
SOURCE_FILES.sort()

#This will be the starting name of the file added to the squad_data_set_directory
game_name = "Elden Ring"
game_being_processed_file_prefix = "elden_ring"
game_being_processed_file_name = f"{squad_data_set_directory}/{game_being_processed_file_prefix}_walkthrough_as_squad"

def load_processed_files_list() -> list[str]:
    try:
        file_to_load = f"{post_processed_directory}/{game_name}.pkl"
        print(f"Loading processed file list from {file_to_load}")
        with open(file_to_load, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Could not find {file_to_load} while loading processed file list.")
        return []

def save_processed_files_list(list_to_save:list[str]):
    file_to_save = f"{post_processed_directory}/{game_name}.pkl" 
    print(f"Saving processed file list to {file_to_save}.")
    with open(file_to_save, 'wb') as f:
        pickle.dump(list_to_save, f)

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

def get_document_contents_from_dir(processed_files_list:list[str]) -> str:
    if not SOURCE_FILES:  # The directory is empty
        print(f"No documents found in {source_directory}")
        return None
    current_file_name = list[str](filter(lambda x: x not in processed_files_list, SOURCE_FILES))[0]
    processed_file_path = os.path.join(source_directory, current_file_name)
    processed_files_list.append(current_file_name)
    print(f"Processing file: {processed_file_path}")
    with open(processed_file_path, 'r') as file:
        content = file.read()
    
    print(f"Found content length: {len(content)} from {processed_file_path}")
    return content
    
def process_document(processed_files_list) -> List[Document]:
    """
    Load document and split in chunks
    """
   
    document_contents = get_document_contents_from_dir(processed_files_list)
    if not document_contents:
        print("Exiting due to no document content found")
        exit(1)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_text(document_contents)
    print(f"Split into {len(texts)} chunks of text (max. {CHUNK_SIZE} tokens each)")
    return texts

def create_ai_gpt3_5_structured_output_chain(): 
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                    openai_api_key=open_api_key,
                    temperature=0)
    
    pydantic_parser = PydanticOutputParser(pydantic_object=SquadDataItem)
    format_instructions = pydantic_parser.get_format_instructions()
    
    template_string = """You are a world class algorithm for extracting question and answer data from user input.
    
    You only return the requested ouput schema and nothing more.

    Take the user input below delimited by triple backticks and use it to create questions and answers.

    user input: ```{user_input}```

    {format_instructions}
    """
    prompt = ChatPromptTemplate(messages=[HumanMessagePromptTemplate.from_template(template_string)],
                                input_variables=["user_input"],
                                partial_variables={"format_instructions": format_instructions},
                                output_parser=pydantic_parser)

    # set verbose=True if you want some debug. Pass it to that function to the left
    return LLMChain(llm=llm, prompt=prompt, output_parser=pydantic_parser, verbose=True), OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm), RetryWithErrorOutputParser.from_llm(parser=pydantic_parser, llm=llm)

# Make the chain
gpt_3_5_chain, gpt_3_5_output_fixing_parser, gpt_3_5_retry_with_error_parser = create_ai_gpt3_5_structured_output_chain()
#Get existing data from last run
existing_squad_data = load_squad_data_from_file()
processed_files = load_processed_files_list()

while files_processed < number_of_files_to_process:
    try:
        documents = process_document(processed_files)  
        for document in documents:
            print("Sending gpt a chunk to process.") 
            try:
                res = gpt_3_5_chain.run(user_input=document)
            except:
                print("GPT sent back incorrect format. Trying the fixing parser.")
                try:
                    res = gpt_3_5_output_fixing_parser.parse(res)
                except:
                    print("Still can't fix it. Retrying as an error.")
                    res = gpt_3_5_retry_with_error_parser.parse_with_prompt(res, gpt_3_5_chain.prompt.format_prompt(user_input=document))
            res.title = game_name
            res.id = str(uuid.uuid4())
            res_asdict = res.dict()               
        files_processed += 1
    except Exception as e:
        print(traceback.format_exc())
        #May need to log whatever existing_squad_data is at the time. It's most likely the reason an exception is thrown here   
        print(f"Exception processing file. Exception: {e}")
        exit(1)

save_squad_data_to_file(existing_squad_data)
save_processed_files_list(processed_files)
print("Done")