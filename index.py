import csv
import sys
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

CHUNK_SIZE = 1000


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


def main():
    embeddings = OllamaEmbeddings(
        # model="jina/jina-embeddings-v2-small-en",
        model="sam860/granite-embedding-multilingual:107m-F16"
    )

    print("Loading CSV...")
    with open(sys.argv[0], "r") as f:
        reader = csv.reader(f)
        texts = [(row[0], row[1]) for row in reader]

    total = len(texts)
    print(f"Loaded ({total} rows)")

    print("Building vectorstore...")
    store = InMemoryVectorStore(embedding=embeddings)

    for n, pairs in enumerate(grouper(texts, CHUNK_SIZE)):
        store.add_documents(
            documents=[Document(page_content=text, id=tid) for tid, text in pairs]
        )
        percent = (n + 1) * CHUNK_SIZE / total * 100
        print(f"Indexed {(n + 1) * CHUNK_SIZE} documents ({percent:0.2f}%)")

    store.dump("vectorstore.json")
    print("Done")


if __name__ == "__main__":
    main()
