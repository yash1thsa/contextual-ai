def upsert_vectors(session, id_vector_pairs):
    from sqlalchemy import text
    import math

    for _id, vec in id_vector_pairs:

        # 1. Validate embedding
        if not vec or len(vec) == 0:
            raise ValueError(f"Empty embedding for id={_id}")

        # 2. Check for NaN or Inf values
        if any(math.isnan(x) or math.isinf(x) for x in vec):
            raise ValueError(f"Invalid numbers (NaN/Inf) in embedding for id={_id}")

        # 3. Convert Python list → pgvector literal
        vec_literal = "[" + ",".join(f"{float(x)}" for x in vec) + "]"

        # 4. Parameterized SQL (safe)
        sql = text("""
            INSERT INTO document_vectors (id, embedding)
            VALUES (:id, :vec::vector)
            ON CONFLICT (id)
            DO UPDATE SET embedding = EXCLUDED.embedding;
        """)

        session.execute(sql, {"id": _id, "vec": vec_literal})
