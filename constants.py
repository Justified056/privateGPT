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
                    },
                    "description": "Answer location in the context. The value is index within the context string. Context index values being at 0. Must be populated"
                },
                "text": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "The answer text from the context property. Must be populated."
                }
            },
            "required": ["answer_start", "text"],
            "description": "Answers for the question and context for the question go here. This is an object not an array."
        },
        "context": {
            "type": "string",
            "description": "Context used to generate the question and answers from. It comes from the user input provided to you. Must be populated."
        },
        "question": {
            "type": "string",
            "description": "Question generated from the context. Must be populated."
        }
    },
    "required": ["answers", "context", "question"]
}

SQUAD_V2_JSON_ARRAY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": {
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
            }
        },
        "required": ["answers", "context", "question"]
    }
}
