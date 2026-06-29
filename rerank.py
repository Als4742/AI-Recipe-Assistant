from sentence_transformers import CrossEncoder
from retrieve import search

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, top_k=15, final_k=5):

    results = search(query)

    docs = results["documents"][0]

    pairs = [(query, doc) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return ranked[:final_k]
