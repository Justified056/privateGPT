import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

SQUAD_V2_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "answers": {
            "type": "object",
            "properties": {
                "answer_start": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    }
                },
                "text": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["answer_start", "text"]
        },
        "context": {
            "type": "string"
        },
        "question": {
            "type": "string"
        },
        "title": {
            "type": "string"
        }
    },
    "required": ["answers", "context", "question"]
}
