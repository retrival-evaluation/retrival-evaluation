from pathlib import Path
import lancedb
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data/raw/scifact"


def load_corpus():
    return pd.read_json(DATA_DIR / "corpus.jsonl", lines=True)

def load_queries():
    return pd.read_json(DATA_DIR / "queries.jsonl", lines=True)

def load_qrels():
    return pd.read_csv(DATA_DIR / "qrels/test.tsv", sep="\t")


def create_corpus_table():
    corpus = load_corpus()
    db = lancedb.connect(DATA_DIR.parent.parent/"lancedb")
    table = db.create_table("corpus", corpus)
    return table