from pathlib import Path
import lancedb
import pandas as pd

# Base directory for all datasets
BASE_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def get_dataset_dir(dataset: str):
    return BASE_DATA_DIR / dataset


def load_corpus(dataset):
    dataset_dir = get_dataset_dir(dataset)
    return pd.read_json(dataset_dir / "corpus.jsonl", lines=True)


def load_queries(dataset):
    dataset_dir = get_dataset_dir(dataset)
    return pd.read_json(dataset_dir / "queries.jsonl", lines=True)


def load_qrels(dataset):
    dataset_dir = get_dataset_dir(dataset)
    return pd.read_csv(dataset_dir / "qrels" / "test.tsv", sep="\t")


def create_corpus_table(dataset):
    corpus = load_corpus(dataset)

    db_path = BASE_DATA_DIR.parent / "lancedb"

    db = lancedb.connect(db_path)

    table_name = f"{dataset}_corpus"

    table = db.create_table(
        table_name,
        corpus,
        mode="overwrite"
    )

    return table
