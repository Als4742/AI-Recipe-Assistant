from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./recipe_db")
collection = client.get_or_create_collection(name="recipes")

def search(query, top_k = 15):

    query_embed = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embed],
        n_results=top_k
    )

    return results


