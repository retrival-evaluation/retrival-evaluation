#Checks for correct answer in results ()
def recall_at_k(results, qrels, k=10):
    qrels = set(qrels)
    top_k_results = set(results[:k])
    
    if len(qrels) == 0:
        return 0.0

    hits = len(top_k_results & qrels)
    return hits / len(qrels)


if __name__ == "__main__":
    print(recall_at_k(
        ["doc1", "doc2", "doc7"],
        {"doc1", "doc7"},
        3
    ))

    print(recall_at_k(
        ["doc2", "doc5", "doc7"],
        {"doc1", "doc7"},
        3
    ))