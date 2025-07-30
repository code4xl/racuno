import fitz  # PyMuPDF for PDFs
import docx
import requests
from bs4 import BeautifulSoup
from tempfile import NamedTemporaryFile

def extract_text_from_url(file_url: str) -> str:
    response = requests.get(file_url)
    ext = file_url.split('?')[0].split('.')[-1].lower()

    with NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
        f.write(response.content)
        f.flush()

        if ext == "pdf":
            doc = fitz.open(f.name)
            return "\n".join([page.get_text() for page in doc])

        elif ext == "docx":
            doc = docx.Document(f.name)
            return "\n".join([p.text for p in doc.paragraphs])

        elif ext == "eml":
            with open(f.name, "r", encoding="utf-8", errors="ignore") as email_file:
                html = email_file.read()
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text(separator="\n")

        else:
            return "❌ Unsupported file format"

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

