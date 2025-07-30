from fastapi import FastAPI, Header, HTTPException
from app.models import QueryRequest, QueryResponse
from app.processor import extract_text_from_url, chunk_text
from app.embedder import embed_chunks
from app.retriever import get_similar_contexts
from app.llm_reasoner import generate_batch_answer  # <-- updated
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

import os
import time
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "FastAPI is live!"}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)

async def run_query(req: QueryRequest):
    start=time.time()
    print(f"📄 Document URL: {req.documents}")
    print(f"❓ Questions: {req.questions}")

    raw_text = extract_text_from_url(req.documents)
    chunks = chunk_text(raw_text)
    db = embed_chunks(chunks)

    batch_size = 5
    answers = []

    for i in range(0, len(req.questions), batch_size):
        question_batch = req.questions[i:i+batch_size]

        # Get context per question
        contexts = [get_similar_contexts(db, q) for q in question_batch]

        # Batch generate answers
        batch_answers = generate_batch_answer(contexts, question_batch)
        answers.extend(batch_answers)
    
    stop=time.time()
    print(stop-start)
    return QueryResponse(answers=answers)
