from sqlalchemy import Table, Column, Integer, String, Text, MetaData, create_engine
from sqlalchemy.sql import func
from sqlalchemy import DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = 'document_chunk'
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_name = Column(String(512), nullable=False)
    page = Column(Integer, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

# Vector table is handled directly by pgvector (we will upsert via SQL)