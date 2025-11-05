import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
from app.services.parser import parse_pdf_to_text_blocks
from app.services.chunker import chunk_text_blocks
from app.services.encoder import encode_chunks
from app.db.transactional_db import SessionLocal
from app.models.models import DocumentChunk
from app.db.vector_store import upsert_vectors
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter()

# Configure logging
logger = logging.getLogger("pdf_upload")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@router.post("/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Received file upload: {file.filename}")

    if not file.filename.lower().endswith(".pdf"):
        logger.warning("Unsupported file type")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save to tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with tmp as f:
        contents = await file.read()
        f.write(contents)
        path = tmp.name
    logger.debug(f"Saved uploaded PDF to temporary path: {path}")

    # 1. Parse PDF into text blocks
    text_blocks = parse_pdf_to_text_blocks(path)
    logger.info(f"Parsed PDF into {len(text_blocks)} text blocks")

    # 2. Chunk blocks
    chunks = chunk_text_blocks(text_blocks)
    logger.info(f"Chunked text into {len(chunks)} chunks")

    # 3. Encode
    embeddings = encode_chunks([c['text'] for c in chunks])
    logger.info(f"Generated embeddings for {len(embeddings)} chunks")

    # 4. Persist: write chunks to Postgres and vectors to pgvector
    session = SessionLocal()
    try:
        saved_ids = []
        for chunk, vector in zip(chunks, embeddings):
            db_chunk = DocumentChunk(
                doc_name=file.filename,
                page=chunk['page'],
                start_char=chunk['start'],
                end_char=chunk['end'],
                text=chunk['text']
            )
            session.add(db_chunk)
            session.flush()  # Get db_chunk.id
            saved_ids.append((db_chunk.id, vector))
            logger.debug(f"Saved chunk id={db_chunk.id} for page={chunk['page']}")

        session.commit()
        logger.info(f"Committed {len(chunks)} chunks to Postgres")

        # Upsert vectors to pgvector table
        upsert_vectors(session, saved_ids)

        session.commit()
        logger.info(f"Upserted {len(saved_ids)} vectors to pgvector")

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()
        logger.debug("Database session closed")

    return JSONResponse({"status": "success", "chunks": len(chunks)})
