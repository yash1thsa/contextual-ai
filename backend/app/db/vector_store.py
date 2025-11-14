import math
from typing import List, Tuple
from sqlalchemy import text

def upsert_vectors(session, id_vector_pairs: List[Tuple[int, List[float]]]):
    """
    Upsert embeddings into pgvector table `document_vectors`.
    Converts numpy arrays → lists → pgvector literal strings.
    """

    sql = text("""
        INSERT INTO document_vectors (id, embedding)
        VALUES (:id, CAST(:vec AS vector))
        ON CONFLICT (id)
        DO UPDATE SET embedding = EXCLUDED.embedding;
    """)

    for _id, vec in id_vector_pairs:

        # ---- 1. Convert numpy array to list ----
        if hasattr(vec, "tolist"):
            vec = vec.tolist()

        # ---- 2. Validate ----
        if not vec or len(vec) == 0:
            raise ValueError(f"Empty embedding for id={_id}")

        if any(math.isnan(x) or math.isinf(x) for x in vec):
            raise ValueError(f"Invalid values (NaN/Inf) in embedding for id={_id}")

        # ---- 3. Convert to pgvector literal "[1,2,3]" ----
        vec_literal = "[" + ",".join(str(float(x)) for x in vec) + "]"

        # ---- 4. Execute SQL ----
        session.execute(sql, {"id": _id, "vec": vec_literal})

    session.commit()
