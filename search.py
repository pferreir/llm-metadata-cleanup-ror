import sys
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = OllamaEmbeddings(
    # model="jina/jina-embeddings-v2-small-en",
    model="sam860/granite-embedding-multilingual:107m-F16"
)

print("Loading vectorstore...")
store = InMemoryVectorStore.load("vectorstore.json", embedding=embeddings)
print("Done")

retriever = store.as_retriever()

with open(sys.argv[1], "r") as f:
    queries = f.readlines()

for query in queries:
    query = query.strip()
    retrieved_documents = retriever.invoke(query)
    if retrieved_documents:
        print(f"{query} -> {retrieved_documents[0].page_content}")
    else:
        print(f"? {query}")
