import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"

PINECONE_INDEX_NAME = "rag-chatbot-v2"
PINECONE_NAMESPACE = "development"

MERGED_DATA_PATH = "data/processed/merged_documents.json"

RETRIEVAL_QUESTIONS_PATH = "evaluation/retrieval_questions.json"

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")