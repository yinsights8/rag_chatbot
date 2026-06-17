from rag.generator import generate_answer, answer_not_found
from rag.retriever import Retriever
from rag.ingestor import parse_and_chunk
from dotenv import load_dotenv
import os

load_dotenv()

datapath = "E:/india_project/rag_aws_/instructions/AWS_Customer_Agreement.pdf"

def main(query: str, pdf_path: str):
    """Run the full RAG pipeline: ingest, retrieve, generate."""
    
    retriever = Retriever()
    chunks = parse_and_chunk(pdf_path)


    chunks = retriever.retrieve(query, top_k=4)
    answer, chunks_ = generate_answer(query, chunks)
    found  = not answer_not_found(answer)
    
    return {
        "query": query,
        "answer": answer,
        "found": found,
        "chunks_": chunks_,
    }


if __name__ == "__main__":  
    query = "How many AWS accounts can a customer create per email address?"
    results = main(query, datapath)
    
    print("Query:", results["query"])
    print("Answer:", results["answer"])
    print("Found:", results["found"])
    print("Chunks:", results["chunks_"])
    