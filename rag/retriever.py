# retriever.py
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
MODEL_NAME = os.getenv("EMBEDDING_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


CHROMA_DIR = f"{DATA_PATH}/chroma_db"


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    def build_index(self, chunks: list[dict]):
        """Embed all chunks and upsert into ChromaDB. Persists automatically."""
        # Clear existing data so re-ingestion is idempotent
        self.client.delete_collection(COLLECTION_NAME)
        # self.collection = self.client.get_or_create_collection(
        #     name=COLLECTION_NAME,
        #     metadata={"hnsw:space": "cosine"},
        # )

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts, show_progress_bar=True, batch_size=32
        ).tolist()

        self.collection.upsert(
            ids=[str(i) for i in range(len(chunks))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{k: v for k, v in c.items() if k != "text"} for c in chunks],
        )

    def load_index(self):
        """Verify the persisted collection exists and has data."""
        count = self.collection.count()
        if count == 0:
            raise FileNotFoundError("No index found. Call POST /ingest first.")

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Return top-k most relevant chunks for a query."""
        q_emb = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = {"text": doc, **meta}
            chunk["score"] = 1 - dist  # cosine distance → similarity
            chunks.append(chunk)

        return chunks