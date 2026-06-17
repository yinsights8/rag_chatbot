from pydantic import BaseModel


class IngestRequest(BaseModel):
    pdf_path: str | None = None


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int


class AskRequest(BaseModel):
    query: str


class SourceChunk(BaseModel):
    text: str
    page: int
    score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    found: bool
    source_chunks: list[SourceChunk]


class FrequentQuestion(BaseModel):
    query: str
    count: int


class AnalyticsResponse(BaseModel):
    most_frequent_questions: list[FrequentQuestion]
    no_answer_queries: list[FrequentQuestion]
    average_latency_ms: float
