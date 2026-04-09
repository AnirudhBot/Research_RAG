import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# App
APP_NAME = "Research Paper Assistant"
APP_VERSION = "2.0.0"

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# LLM
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0.2

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"

# Vector store
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 5

# Image extraction
MIN_IMAGE_DIMENSION = 100  # pixels — skip icons/logos smaller than this
IMAGE_DESCRIPTION_MODEL = os.getenv("IMAGE_DESCRIPTION_MODEL", "gpt-4o")

# UI
PAGE_TITLE = "Research Paper Assistant"
PAGE_LAYOUT = "wide"
