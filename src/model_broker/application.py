"""FastAPI entry point for model-broker.

Run with ``uvicorn --factory model_broker.application:create_app``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class Settings:
    """Deployment configuration read from the environment."""

    llama_url: str

    @classmethod
    def from_environment(cls) -> Settings:
        llama_url = os.environ.get("MODEL_BROKER_LLAMA_URL", "http://llama-cpp:8080")
        return cls(llama_url=llama_url.rstrip("/"))


def not_implemented(message: str, code: str) -> JSONResponse:
    """Return a 501 response in the OpenAI error shape."""
    error = {"message": message, "type": "not_implemented", "code": code}
    return JSONResponse(status_code=501, content={"error": error})


router = APIRouter(tags=["broker"])


@router.get("/health")
async def health() -> JsonObject:
    """Report broker liveness."""
    return {"status": "ok"}


@router.post(
    "/v1/chat/completions",
    status_code=501,
    summary="Broker-owned chat completions",
    responses={501: {"description": "Scheduling and proxying are not implemented yet."}},
)
async def chat_completions() -> JSONResponse:
    """Reserve the chat endpoint for validation, scheduling, and proxying."""
    return not_implemented(
        "Chat completions are broker-owned and require scheduling support that is not "
        "implemented yet.",
        "broker_scheduling_not_implemented",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the broker API. Paths the broker does not serve return 404."""
    app = FastAPI(
        title="Model broker",
        version="0.1.0",
        description="Broker-owned API surface for scheduled local model inference.",
    )
    app.state.settings = settings or Settings.from_environment()
    app.include_router(router)
    return app
