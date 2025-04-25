import hashlib
import random
import string

def get_document_id(text: str) -> str:
    """Generate a unique document ID based on content."""
    return hashlib.sha256(text.encode()).hexdigest()

def generate_collection_name(prefix: str = "research_papers") -> str:
    """Generate a random collection name for Qdrant."""
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}_{random_suffix}"

def format_error_message(error: Exception) -> str:
    """Format error messages for UI display."""
    return f"An error occurred: {str(error)}" 