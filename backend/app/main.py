from fastapi import FastAPI
from app.routes import query, upload

app = FastAPI()

app.include_router(upload.router, prefix="/upload")
app.include_router(query.router, prefix="/query")

@app.get("/")
async def root():
    return {"status": "ok", "service": "pdf-ingest"}