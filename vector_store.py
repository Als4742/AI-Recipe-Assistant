from sentence_transformers import SentenceTransformer
import chromadb
from prepare_data import load_recipes, build_doc

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./recipe_db")
collection = client.get_or_create_collection(name="recipes")


def embed_and_store():
    df = load_recipes()
    df = df.sample(10000)
    docs = build_doc(df)

    print("Encoding embeddings in batch...")

    batch_size = 20

    for i in range(0, len(docs), batch_size):

        batch_docs = docs[i:i+batch_size]

        embeddings = model.encode(batch_docs, show_progress_bar=True).tolist()

        ids = [str(j) for j in range(i, i+len(batch_docs))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=batch_docs
        )

if __name__ == "__main__":
    embed_and_store()