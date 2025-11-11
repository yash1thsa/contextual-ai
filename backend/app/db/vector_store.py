def upsert_vectors(session, id_vector_pairs):
    from sqlalchemy import text

    for _id, vec in id_vector_pairs:
        # Convert vector to literal string for pgvector
        vec_literal = "[" + ",".join(map(str, vec)) + "]"
        sql = f"""
            INSERT INTO document_vectors (id, embedding)
            VALUES ({_id}, '{vec_literal}'::vector)
            ON CONFLICT (id)
            DO UPDATE SET embedding = EXCLUDED.embedding;
        """
        session.execute(text(sql))
