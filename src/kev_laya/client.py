"""Typed local client using the frozen native System One wire shapes."""
from __future__ import annotations
from typing import Any
import httpx
from .schema import Question, SystemOneRequest, SystemOneResponse


class KevLayaClient:
    def __init__(self, base_url="http://127.0.0.1:8009", api_key: str | None = None,
                 model="kev-laya-preview", timeout=120.0, *, transport=None):
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)
        self.model = model
    def system_one(self, *, state: Any, questions: dict[str, Question | dict], model: str | None = None) -> SystemOneResponse:
        request = SystemOneRequest(state=state, questions=questions, model=model or self.model)
        response = self.http.post("/v1/systemone", json=request.model_dump())
        response.raise_for_status()
        return SystemOneResponse.model_validate(response.json())
    def models(self) -> dict:
        response = self.http.get("/v1/models")
        response.raise_for_status()
        return response.json()
    def close(self):
        self.http.close()
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()
