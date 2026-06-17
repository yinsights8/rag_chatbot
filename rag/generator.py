# generator.py

from openai import OpenAI
import os 
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# create a client for the generator model
def model_client():
    """Return an OpenAI client for the generator model."""
    
    client = OpenAI(
            base_url=BASE_URL,
            api_key=OPENAI_API_KEY,
        )

    return client



SYSTEM_PROMPT = """You are a precise document assistant. Answer questions ONLY using the
provided context excerpts. If the answer cannot be found in the context, respond with exactly:
"I cannot find the answer to this question in the provided document."
Do not make up facts. Do not use outside knowledge."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Construct a grounded RAG prompt from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[Source {i} — Page {chunk['page']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

def answer_not_found(response: str) -> bool:
    """Detect if the LLM indicated it could not find an answer."""
    markers = [
        "cannot find", "not found in the document", "not mentioned",
        "no information", "I don't know", "not in the provided",
    ]
    return any(m.lower() in response.lower() for m in markers)


def generate_answer(query: str, chunks: list[dict]) -> str:
    """Run the full generation step and return the answer string."""
    client = model_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, chunks)},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip(), chunks