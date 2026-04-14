import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
      result = client.models.embed_content(
          model="gemini-embedding-001",
          contents=texts,
          config={"task_type": task_type},
      )
      return [e.values for e in result.embeddings]

def embed_query(query):
      result = client.models.embed_content(
          model="text-embedding-004",
          contents=query,
          config={"task_type": "RETRIEVAL_QUERY"},
      )
      return result.embeddings[0].values