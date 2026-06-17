import os
import time
from fastapi import FastAPI, HTTPException
from rag.schema import IngestRequest, IngestResponse, AskRequest, AskResponse, AnalyticsResponse

from rag.ingestor import parse_and_chunk
from rag.retriever import Retriever
from rag.generator import generate_answer, answer_not_found
# from rag.db import (
#     init_db,
#     log_query,
#     get_most_frequent_questions,
#     get_no_answer_queries,
#     get_average_latency_ms,
# )

DEFAULT_PDF_PATH = "instructions/AWS_Customer_Agreement.pdf"

app = FastAPI(title="RAG Document Q&A API")
retriever = Retriever()

# @app.on_event("startup")
# def on_startup():
#     init_db()


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    pdf_path = request.pdf_path or DEFAULT_PDF_PATH
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF not found at path: {pdf_path}")

    try:
        chunks = parse_and_chunk(pdf_path)
        retriever.build_index(chunks)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Ingestion failed due to an internal error.")

    return IngestResponse(message=f"Indexed {len(chunks)} chunks from {pdf_path}", chunks_indexed=len(chunks))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        retriever.load_index()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="No document has been ingested yet. Call POST /ingest first.")

    try:
        start = time.perf_counter()
        chunks = retriever.retrieve(request.query, top_k=4)
        answer = generate_answer(request.query, chunks)
        latency_ms = (time.perf_counter() - start) * 1000

        found = not answer_not_found(answer)
        top_score = max((c["score"] for c in chunks), default=None)
        # log_query(request.query, answer, found, top_score, latency_ms)

        source_chunks = [{"text": c["text"], "page": c["page"], "score": c["score"]} for c in chunks]
        return AskResponse(query=request.query, answer=answer, found=found, source_chunks=source_chunks)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.")


# @app.get("/analytics", response_model=AnalyticsResponse)
# def analytics():
#     try:
#         return AnalyticsResponse(
#             most_frequent_questions=get_most_frequent_questions(),
#             no_answer_queries=get_no_answer_queries(),
#             average_latency_ms=get_average_latency_ms(),
#         )
#     except Exception:
#         raise HTTPException(status_code=500, detail="Failed to compute analytics.")
