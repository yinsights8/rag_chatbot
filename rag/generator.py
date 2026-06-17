# generator.py

from openai import OpenAI
import re
import os 
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
RETRIEVAL_CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD"))

NOT_FOUND_MESSAGE = "I cannot find the answer to this question in the provided document."

# create a client for the generator model
def model_client():
    """Return an OpenAI client for the generator model."""
    
    client = OpenAI(
            base_url=BASE_URL,
            api_key=OPENAI_API_KEY,
        )

    return client



SOURCES_USED_MARKER = "SOURCES_USED:"

SYSTEM_PROMPT = f"""You are a precise document assistant. Answer questions ONLY using the
provided context excerpts. If the answer cannot be found in the context, respond with exactly:
"{NOT_FOUND_MESSAGE}"
Do not make up facts. Do not use outside knowledge.

After your answer, on a new line, write exactly which numbered sources you relied on to
construct the answer, in this format: {SOURCES_USED_MARKER} 1,3
If you could not answer from the context, write: {SOURCES_USED_MARKER} none"""



def build_prompt(query: str, chunks: list[dict]) -> str:
    """Construct a grounded RAG prompt from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[Source {i} — Page {chunk['page']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

def answer_not_found(response: str) -> bool:
    """Detect if the LLM indicated it could not find an answer."""
    if response.strip() == NOT_FOUND_MESSAGE:
        return True
    markers = ["cannot find the answer", "not found in the provided document"]
    return any(m.lower() in response.lower() for m in markers)


# split the model's SOURCES_USED citation line out of the answer text
def parse_sources_used(raw_response: str, num_chunks: int) -> tuple[str, list[int]]:
      """
      Returns the cleaned answer (citation line removed) and the 1-based source
      indices the model claims it used, filtered to the valid range.
      """
      matches = list(re.finditer(rf"^\s*{re.escape(SOURCES_USED_MARKER)}\s*(.*)$",
  raw_response, re.IGNORECASE | re.MULTILINE))
      if not matches:
          return raw_response.strip(), []
      match = matches[-1]

      clean_answer = raw_response[: match.start()].strip()
      value = match.group(1).strip().lower()
      if value == "none" or not value:
          return clean_answer, []

      indices = []
      for part in value.split(","):
          part = part.strip()
          if part.isdigit():
              index = int(part)
              if 1 <= index <= num_chunks and index not in indices:
                  indices.append(index)
      return clean_answer, indices




def generate_answer(query: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Run the full generation step.
    Returns the answer text and the chunk(s) the model actually cited as used.
    """
    if not chunks or max(c["score"] for c in chunks) < RETRIEVAL_CONFIDENCE_THRESHOLD:
        return NOT_FOUND_MESSAGE, []

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
    raw = response.choices[0].message.content.strip()
    clean_answer, indices = parse_sources_used(raw, len(chunks))

    if answer_not_found(clean_answer):
        return clean_answer, []
    if indices:
        return clean_answer, [chunks[i - 1] for i in indices]
    # Model didn't follow the citation format — fall back to the single
    # highest-scoring retrieved chunk.
    return clean_answer, [max(chunks, key=lambda c: c["score"])]