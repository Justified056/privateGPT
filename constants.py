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
        "data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
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
                                            "answers": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "answer_start": {
                                                            "type": "integer"
                                                        },
                                                        "text": {
                                                            "type": "string"
                                                        }
                                                    },
                                                    "required": [
                                                        "text",
                                                        "answer_start"
                                                    ]
                                                }
                                            },
                                            "id": {
                                                "type": "string"
                                            },
                                            "plausible_answers": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "answer_start": {
                                                            "type": "integer"
                                                        },
                                                        "text": {
                                                            "type": "string"
                                                        }
                                                    },
                                                    "required": [
                                                        "text",
                                                        "answer_start"
                                                    ]
                                                }
                                            },
                                            "question": {
                                                "type": "string"
                                            }
                                        },
                                        "required": [
                                            "id",
                                            "question",
                                            "answers"
                                        ]
                                    }
                                }
                            },
                            "required": [
                                "context",
                                "qas"
                            ]
                        }
                    },
                    "title": {
                        "type": "string"
                    }
                },
                "required": [
                    "title",
                    "paragraphs"
                ]
            }
        },
        "version": {
            "type": "string"
        }
    },
    "required": [
        "data",
        "version"
    ]
}
