import lancedb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from exjobb_rag.data_loader import load_corpus

# Path to local lancedb database
DB_PATH = Path(__file__).parent.parent.parent / "data" / "lancedb"

EMBED_MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5")

def embed_texts(texts, task_type=None):
    embeddings = EMBED_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()

def embed_query(query):
    return EMBED_MODEL.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        convert_to_numpy=True
    ).tolist()

def embed_corpus(batch_size=100, limit=None):
    # Embed all corpus documents and store them in LanceDB.
    
    corpus = load_corpus()
    if limit is not None:
        corpus = corpus.head(limit).copy()

    # Combine title + text for each document gives richer embeddings
    # than using just the abstract text alone
    combined = (corpus["title"] + " " + corpus["text"]).tolist()

    # Process in batches of 100 for memory efficiency and progress reporting
    
    all_embeddings = []
    for i in range(0, len(combined), batch_size):
        batch = combined[i:i + batch_size]
        print(f"Batch {i // batch_size + 1}/{(len(combined) + batch_size - 1) // batch_size}")
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)
        

    # Add the embeddings vectors as a new column in the DataFrame
    corpus["vector"] = all_embeddings

    # Save to LanceDB  it automatically recognizes the "vector" column
    # as embeddings and builds a search index from it
    db = lancedb.connect(DB_PATH)
    table = db.create_table("corpus_embeddings", corpus, mode="overwrite")
    print(f"Done! Stored {len(corpus)} documents with embeddings in LanceDB.")
    return table


