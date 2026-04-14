import os
from google import genai
from dotenv import load_dotenv
import time
import lancedb
from pathlib import Path
from exjobb_rag.data_loader import load_corpus

# Path to local lancedb database
DB_PATH = Path(__file__).parent.parent.parent / "data" / "lancedb"

# Configure Gemini API client
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"

def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
      # Embed a list of texts. Returns a list of vectors
      result = client.models.embed_content(
          model=EMBED_MODEL,
          contents=texts,
          config={"task_type": task_type},
      )
      return [e.values for e in result.embeddings]

def embed_query(query):
      # Embed a single search query. Uses RETRIEVAL_QUERY for optimized search.
      result = client.models.embed_content(
          model=EMBED_MODEL,
          contents=query,
          config={"task_type": "RETRIEVAL_QUERY"},
      )
      return result.embeddings[0].values

def embed_corpus(batch_size=100):
    # Embed all corpus documents and store them in LanceDB.
    
    corpus = load_corpus()

    # Combine title + text for each document — gives richer embeddings
    # than using just the abstract text alone
    combined = (corpus["title"] + " " + corpus["text"]).tolist()

    # We can't send all 5183 texts at once — the API has limits.
    # So we process in batches of 100 at a time.
    all_embeddings = []
    for i in range(0, len(combined), batch_size):
        batch = combined[i:i + batch_size]
        print(f"Batch {i // batch_size + 1}/{(len(combined) + batch_size - 1) // batch_size}")
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)
        # Small pause between batches to avoid hitting API rate limits
        time.sleep(1)

    # Add the embeddings vectors as a new column in the DataFrame
    corpus["vector"] = all_embeddings

    # Save to LanceDB — it automatically recognizes the "vector" column
    # as embeddings and builds a search index from it
    db = lancedb.connect(DB_PATH)
    table = db.create_table("corpus_embeddings", corpus, mode="overwrite")
    print(f"Done! Stored {len(corpus)} documents with embeddings in LanceDB.")
    return table


