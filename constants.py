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

SQUAD_V2_JSON_SCHEMA = """{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string"
                    },
                    "paragraphs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "context": {
                                    "type": "string"
                                },
                                "qas": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {
                                                "type": "string"
                                            },
                                            "question": {
                                                "type": "string"
                                            },
                                            "answers": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "text": {
                                                            "type": "string"
                                                        },
                                                        "answer_start": {
                                                            "type": "integer"
                                                        }
                                                    },
                                                    "required": ["text", "answer_start"]
                                                }
                                            },
                                            "is_impossible": {
                                                "type": "boolean"
                                            },
                                            "plausible_answers": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "text": {
                                                            "type": "string"
                                                        },
                                                        "answer_start": {
                                                            "type": "integer"
                                                        }
                                                    },
                                                    "required": ["text", "answer_start"]
                                                }
                                            }
                                        },
                                        "required": ["id", "question", "answers", "is_impossible"]
                                    }
                                }
                            },
                            "required": ["context", "qas"]
                        }
                    }
                },
                "required": ["title", "paragraphs"]
            }
        },
        "version": {
            "type": "string"
        }
    },
    "required": ["data", "version"]
}"""
