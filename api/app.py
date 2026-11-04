import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager

from retrieval.recursive_retrieval import create_recursive_retriever
from generation.generation import answer_question, configure_mlflow
from api.usage_limits import check_usage_limit


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://harrisonfunk2.github.io"
    ],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

class ChatResponse(BaseModel):
    answer: str
    citations: dict[int, str]


@app.get("/health")
def health():
    return {"status": "ok"}

logger = logging.getLogger(__name__)



@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    # Emergency switch
    if os.getenv("CHAT_ENABLED", "false").lower() != "true":
        raise HTTPException(
            status_code=503,
            detail="Chatbot is temporarily unavailable."
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty."
        )

    retriever = getattr(app.state, "retriever", None)

    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot is not ready."
        )

    # Check the shared quota before any OpenAI calls.
    try:
        allowed = check_usage_limit()

    except Exception:
        logger.exception("Firestore usage-limit check failed")
        raise HTTPException(
            status_code=503,
            detail="Chatbot is temporarily unavailable."
        )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Chatbot usage limit reached. Please try later.",
            headers={"Retry-After": "3600"}
        )

    try:
        output = answer_question(question, retriever)

        return ChatResponse(
            answer=output["answer"],
            citations=output["citation_map"]
        )

    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer right now."
        )
