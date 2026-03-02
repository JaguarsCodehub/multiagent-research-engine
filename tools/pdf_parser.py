import fitz  # PyMuPDF
import os

def parse_document(file_path: str) -> str:
    """
    Parse text from a local file path (PDF, TXT, MD, etc.).
    """
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found."
        
    try:
        if file_path.lower().endswith(('.txt', '.md', '.csv', '.json', '.py', '.js')):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
    except Exception as e:
        return f"Error parsing document {file_path}: {e}"

def chunk_document(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Split document text into overlapping chunks using word boundary approx.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks
