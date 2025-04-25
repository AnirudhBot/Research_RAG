import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# App settings
APP_NAME = "Research Paper Assistant"
APP_VERSION = "1.0.0"

# API Keys and External Services
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")

# Vector Store Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# UI Settings
PAGE_TITLE = "Research Paper Assistant"
PAGE_LAYOUT = "wide" 