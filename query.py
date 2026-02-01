from pinecone import Pinecone
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("developer-quickstart-py")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

query = input("Ask: ")
res = index.search(
    namespace="__default__",
    query={
        "inputs": {"text": query},
        "top_k": 10,
    }
)
""" 
If in some cases model replies no data found and that data is in the data folder you can 
check the retrieved chunks and tell what the actual issue is
print("\nRetrieved chunks:\n")

for i, hit in enumerate(res["result"]["hits"]):
    print(f"--- Match {i+1} ---")
    print(hit["fields"]["chunk_text"])
    print()"""


context = "\n\n".join(
    [hit["fields"]["chunk_text"] for hit in res["result"]["hits"]]
)


prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "Not found in the data".

Context:
{context}

Question:
{query}

Answer:
"""

chat = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role":"user","content":prompt}]
)


print("\nFinal Answer:\n")
print(chat.choices[0].message.content)
