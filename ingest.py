from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import time
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "developer-quickstart-py"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )
    while not pc.describe(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
with open("data/sample.txt" , "r" ,encoding = "utf-8") as f:
    text = f.read()

split = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)

chunk = split.split_text(text)

dbvect = []
for i,c in enumerate(chunk):
    dbvect.append({
        "id" : f"chunk-{i}",
        "chunk_text" : c,
        "source" : "sample.txt",
        "iid" : i
    })

index.upsert_records(namespace="__default__",records=dbvect)
print("Data ingested")