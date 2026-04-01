# retrival-evaluation
Benchmarking classic, semantic, and hybrid search methods using the BEIR framework. Evaluating Recall@k, MRR, nDCG, and latency for vector databases. This is a thesis by: Casper Zanichelli &amp; Jonas Gustafsson 2026

## Dataset: SciFact

We use the [SciFact](https://github.com/allenai/scifact) dataset from the BEIR benchmark. The data is located in `data/raw/scifact/` and consists of three parts:

### Corpus (`corpus.jsonl`)
The corpus contains 5,183 scientific abstracts. This is the "search database" — all documents that the search methods will retrieve from. Each entry has:
- `_id` — unique document identifier
- `title` — title of the paper
- `text` — the full abstract text

### Queries (`queries.jsonl`)
The queries contain 1,109 scientific claims/statements that will be used to test the search methods. Each entry has:
- `_id` — unique query identifier
- `text` — a scientific claim (e.g. "0-dimensional biomaterials lack inductive properties")

### Relevance Judgments / Qrels (`qrels/test.tsv`)
The qrels (query relevance judgments) are the ground truth — they tell us which documents are the correct answer for a given query. The test set contains **339 query-document pairs**. Each entry has:
- `query-id` — maps to a query `_id`
- `corpus-id` — maps to a corpus `_id`
- `score` — relevance score (1 = relevant)

Without ground truth, we would have no way of measuring whether a search method is good or bad. This is what makes BEIR datasets valuable — someone has manually reviewed and labeled which queries should match which documents.

### How evaluation works
The flow is simple: give a search method a query → it returns ranked documents → compare against the ground truth (qrels) → measure performance using metrics like Recall@k, MRR, and nDCG.
