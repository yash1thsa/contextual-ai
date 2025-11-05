from fastapi import FastAPI
from app.routes.upload import router as upload_router


app = FastAPI(title="PDF Ingest Service")


app.include_router(upload_router, prefix="/upload", tags=["upload"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "pdf-ingest"}