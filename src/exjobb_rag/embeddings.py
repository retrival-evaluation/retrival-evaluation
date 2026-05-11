import lancedb
from pathlib import Path
from sentence_transformers import SentenceTransformer

from exjobb_rag.data_loader import load_corpus

# Path to local lancedb database
DB_PATH = Path(__file__).parent.parent.parent / "data" / "lancedb"

# Load embedding model
EMBED_MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5")

def embed_texts(texts, task_type=None):
    embeddings = EMBED_MODEL.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings.tolist()

def embed_corpus(dataset, batch_size=100, limit=None):
    corpus = load_corpus(dataset)

    if limit is not None:
        corpus = corpus.head(limit).copy()

    combined = (corpus["title"] + " " + corpus["text"]).tolist()

    all_embeddings = []

    for i in range(0, len(combined), batch_size):
        batch = combined[i:i + batch_size]
        print(f"Batch {i // batch_size + 1}/{(len(combined) + batch_size - 1) // batch_size}")

        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)

    corpus["vector"] = all_embeddings

    db = lancedb.connect(DB_PATH)
    table = db.create_table(
        f"{dataset}_corpus_embeddings",
        corpus,
        mode="overwrite"
    )

    print(f"Done! Stored {len(corpus)} documents from {dataset} with embeddings in LanceDB.")
    return table


if __name__ == "__main__":
    embed_corpus(dataset="trec-covid")