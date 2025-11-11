# app/routes/2_query.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.db.transactional_db import SessionLocal
from app.services.encoder import encode_chunks
from app.services.retriever import retrieve_similar_chunks
from app.services.llm import answer_with_context
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@router.post("/")
async def query_documents(req: QueryRequest):
    session = SessionLocal()
    try:
        # Step 1: embed the question
        embedding = encode_chunks([req.question])[0]

        # Step 2: retrieve relevant chunks
        chunks = retrieve_similar_chunks(session, embedding, top_k=req.top_k)
        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant chunks found")

        # Step 3: use LLM to answer
        answer = answer_with_context(req.question, chunks)

        logger.info(f"Answer generated for query: {req.question}")
        return {"query": req.question, "answer": answer, "context": chunks}

    finally:
        session.close()
