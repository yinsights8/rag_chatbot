from rag.generator import generate_answer, answer_not_found
from rag.retriever import Retriever
from rag.ingestor import parse_and_chunk
from dotenv import load_dotenv
import os

load_dotenv()

datapath = "E:/india_project/rag-aws/instructions/AWS_Customer_Agreement.pdf"


def main(query: str, pdf_path: str):
    """Run the full RAG pipeline: ingest, retrieve, generate."""

    retriever = Retriever()
    parsed_chunks = parse_and_chunk(pdf_path)
    retriever.build_index(parsed_chunks)

    retrieved_chunks = retriever.retrieve(query, top_k=5)
    answer = generate_answer(query, retrieved_chunks)
    found = not answer_not_found(answer)

    source_chunks = [
        {"text": c["text"], "page": c["page"], "score": c["score"]}
        for c in retrieved_chunks
    ]

    return {
        "query": query,
        "answer": answer,
        "found": found,
        "source_chunks": source_chunks,
    }


if __name__ == "__main__":
    query = "who is batman?"
    results = main(query, datapath)

    print("Query:", results["query"])
    print("Answer:", results["answer"])
    print("Found:", results["found"])
    print("Source chunks:")
    for chunk in results["source_chunks"]:
        print(f"  page {chunk['page']} (score={chunk['score']:.3f}): {chunk['text'][:100]}")
