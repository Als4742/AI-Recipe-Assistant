from rerank import rerank
from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")

SYSTEM_PROMPT = """
  You are a grounded recipe assistant.

RULES:
- Use ONLY the provided recipes.
- Do NOT add external knowledge.
- If information is missing, say "Not found in provided recipes."
- Keep response short and structured.
- Always mention recipe names as sources.
- If some of information is missing, connect dots and give response.
"""

def build_context(ranked_results):
    context = ""

    for i, (doc, score) in enumerate(ranked_results):

        context += f"\n\n[Recipe {i+1} | Score: {score:.2f}]\n"
        context += doc

    return context

def call_llm(query, context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, 
        {"role": "user", "content": f"""Context information is provided below. 

Review the context chunks and answer the question. 
         
Question: {query}

Context:
{context}

Final Answer:"""}
    ]

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            messages=messages,
            temperature=0, 
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n[API Error]: {e}")
        return None


def rag(query):
    ranked = rerank(query)
    context = build_context(ranked)
    answer = call_llm(query, context)

    return answer

if __name__ == "__main__":

    while True:
        print("="*50)
        q = input("\nAsk recipe question (or exit): ")

        if q.lower() == "exit":
            break

        result = rag(q)

        print("\n\n===== FINAL ANSWER =====\n")
        print(result)