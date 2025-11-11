# app/services/retriever.py
from sqlalchemy import text

def retrieve_similar_chunks(session, query_embedding, top_k=5):
    """
    Returns top-k most similar chunks to a given embedding.
    """
    vec_literal = "[" + ",".join(map(str, query_embedding)) + "]"

    sql = text(f"""
        SELECT dc.id, dc.text, dc.page, 
               1 - (embedding <=> '{vec_literal}'::vector) AS similarity
        FROM document_vectors dv
        JOIN document_chunk dc ON dv.id = dc.id
        ORDER BY embedding <=> '{vec_literal}'::vector
        LIMIT :top_k;
    """)
    result = session.execute(sql, {"top_k": top_k})
    rows = result.fetchall()
    return [{"id": r.id, "page": r.page, "text": r.text, "similarity": r.similarity} for r in rows]
