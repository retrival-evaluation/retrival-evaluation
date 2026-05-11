import lancedb
from pathlib import Path
from exjobb_rag.embeddings import embed_query

DB_PATH = Path(__file__).parent.parent.parent / "data" / "lancedb"

def semantic_search(query: str, k: int = 10):
    """
    Find the top-k documents most semantically similar to the query.
   
    Returns a pandas DataFrame with columns: _id, title, text, _distance.
    _distance is cosine distance (lower = more similar).
    """
    
    # Embed the query (BGE prefix applied inside embed_query)
    query_vec = embed_query(query)

    # Open the LanceDB table
    db = lancedb.connect(DB_PATH)
    table = db.open_table("corpus_embeddings")

    # Search using cosine similarity BGE models are trained for it

    results = (
        table.search(query_vec)
        .metric("cosine")
        .limit(k)
        .to_pandas()
      )

    return results