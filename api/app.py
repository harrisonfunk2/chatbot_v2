from fastapi import FastAPI
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager

from retrieval.recursive_retrieval import create_recursive_retriever
from generation.generation import answer_question, configure_mlflow


# lifespan function runs when the server starts and shuts down
@asynccontextmanager
async def lifespan(app: FastAPI):

    configure_mlflow()

    app.state.retriever = create_recursive_retriever(
        summary_top_k=2,
        page_top_k=3
    )

    yield

    app.state.retriever = None


app = FastAPI(title="Chatbot",
              lifespan=lifespan)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

class ChatResponse(BaseModel):
    answer: str
    citations: dict[int, str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    retriever = app.state.retriever

    output = answer_question(
        request.question,
        retriever
    )

    return ChatResponse(
        answer=output["answer"],
        citations=output["citation_map"]
    )