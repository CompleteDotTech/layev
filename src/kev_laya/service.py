"""Non-generative decision serving with bounded request-local state and quotas."""
from __future__ import annotations
from dataclasses import dataclass
import hmac
import math
import threading
import time
from typing import Callable
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from .encoding import ContextOverflow, Limits, encode_request
from .execution import BatchBudgetExceeded
from .schema import (ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneRequest, SystemOneResponse,
                     Usage, canonical, render, strict_loads)
from .telemetry import ServingHook


@dataclass(frozen=True)
class ServeSettings:
    api_keys: tuple[str, ...] = ()
    allow_unauthenticated: bool = False
    max_concurrency: int = 1
    requests_per_minute: int = 120
    input_tokens_per_minute: int = 262144
    max_body_bytes: int = 2097152
    retention_disabled: bool = True
    def __post_init__(self):
        if min(self.max_concurrency, self.requests_per_minute, self.input_tokens_per_minute, self.max_body_bytes) < 1:
            raise ValueError("invalid serving bounds")
        if len(self.api_keys) > 64 or len(set(self.api_keys)) != len(self.api_keys) or any(not isinstance(k, str) or not k for k in self.api_keys):
            raise ValueError("invalid key configuration")
        if not self.api_keys and not self.allow_unauthenticated:
            raise ValueError("configure API keys or explicit loopback-only unauthenticated mode")
        if not self.retention_disabled:
            raise ValueError("request retention is not implemented; diagnostic export must be separate and explicit")


class Quotas:
    def __init__(self, settings: ServeSettings, clock: Callable[[], float] = time.monotonic):
        self.settings, self.clock = settings, clock
        self._lock = threading.Lock()
        self.buckets: dict[int, tuple[float, float, float]] = {}
    def admit(self, principal: int, tokens: int):
        with self._lock:
            now = self.clock()
            rpm, tpm = self.settings.requests_per_minute, self.settings.input_tokens_per_minute
            r, t, last = self.buckets.get(principal, (float(rpm), float(tpm), now))
            elapsed = max(0., now - last)
            r, t = min(rpm, r + elapsed * rpm / 60), min(tpm, t + elapsed * tpm / 60)
            self.buckets[principal] = (r, t, now)
            if r < 1 or t < tokens:
                delay = max((1 - r) * 60 / rpm, (tokens - t) * 60 / tpm, 1)
                raise HTTPException(429, {"code": "quota_exceeded"}, headers={"Retry-After": str(math.ceil(delay))})
            self.buckets[principal] = (r - 1, t - tokens, now)


class InferenceRuntime:
    def __init__(self, model, tokenizer, model_id: str, limits: Limits, *, stable=False, allow_untrained=False):
        if model.training_steps < 1 and not allow_untrained:
            raise ValueError("refusing to serve an untrained decision head")
        if not model_id.startswith("kev-laya-"):
            raise ValueError("replacement must report its own model identity")
        if limits.branch > model.cfg.max_position_embeddings:
            raise ValueError("serving limit exceeds the actual backbone window")
        self.model = model.eval()
        self.tokenizer, self.model_id, self.limits = tokenizer, model_id, limits
        self.aliases = {model_id: model_id, "kev-laya-preview": model_id}
        if stable:
            self.aliases["kev-laya-latest"] = model_id

    def models(self) -> dict:
        return {"models": [{"name": name, "description": "Independent Kev-Laya research candidate; see acceptance evidence",
                            "release_date": "2026-09-25"} for name in self.aliases],
                "resolved_model": self.model_id,
                "context_limits": {"branch": self.limits.branch, "aggregate": self.limits.aggregate},
                "native_weights_loaded": self.model.native_weights_loaded,
                "confidence_definition": "entropy-concentration-v1",
                "question_batching": self.model.batch_policy.to_dict(),
                "input_serialization": __import__("kev_laya.encoding", fromlist=["preprocessing_identity"]).preprocessing_identity(self.tokenizer),
                "legacy_lossy_preprocessing": getattr(self.tokenizer, "literal_encoding", None) == "special-spelling-v1"}

    def encode(self, request: SystemOneRequest):
        if request.model not in self.aliases:
            raise HTTPException(422, {"code": "unknown_model", "field": "model"})
        return encode_request(request, self.tokenizer, self.limits)

    @torch.inference_mode()
    def predict_encoded(self, encoding) -> tuple[SystemOneResponse, dict]:
        logits, compute = self.model(encoding)
        answers = {}
        for branch, z in zip(encoding.branches, logits, strict=True):
            p = (z.double() / self.model.temperatures[branch.question.type]).softmax(-1).cpu().tolist()
            if not all(math.isfinite(v) for v in p):
                raise FloatingPointError("nonfinite model output")
            confidence = 1.0 if len(p) == 1 else max(0.0, min(1.0, 1 + sum(v * math.log(v) for v in p if v > 0) / math.log(len(p))))
            keys = [key for key, _ in branch.question.options()]
            probabilities = dict(zip(keys, p, strict=True))
            if branch.question.type == "choice":
                ans = ChoiceAnswer(choice=keys[max(range(len(p)), key=p.__getitem__)], confidence=confidence, probabilities=probabilities)
            elif branch.question.type == "score":
                ans = ScoreAnswer(score=sum(i * v for i, v in enumerate(p)), confidence=confidence,
                                  probabilities=probabilities, legend={str(i): render(v) for i, v in enumerate(branch.question.criteria)})
            else:
                ans = NoulAnswer(noul=p[1])
            answers[branch.question_id] = ans
        output_json = canonical({k: v.model_dump() for k, v in answers.items()})
        output_encode = getattr(self.tokenizer, "encode_output", self.tokenizer.encode)
        response = SystemOneResponse(model=self.model_id, answers=answers,
                     usage=Usage(input_tokens=encoding.logical_tokens, output_tokens=len(output_encode(output_json)),
                                 forward_tokens=compute["forward_tokens"], state_tokens=len(encoding.state),
                                 branch_tokens=[len(b.ids) for b in encoding.branches]),
                     calibration_status=self.model.calibration_provenance["status"])
        return response, compute


def create_app(runtime: InferenceRuntime, settings: ServeSettings, hook: ServingHook | None = None) -> FastAPI:
    app = FastAPI(title="Kev-Laya", version="0.1.0")
    slots = threading.BoundedSemaphore(settings.max_concurrency)
    quotas = Quotas(settings)
    hook = hook or ServingHook()
    app.state.runtime, app.state.slots, app.state.quotas, app.state.hook = runtime, slots, quotas, hook

    def authenticate(request: Request) -> int:
        header = request.headers.get("authorization", "")
        candidate = header[7:] if header.startswith("Bearer ") else ""
        found = None
        for i, key in enumerate(settings.api_keys):
            if hmac.compare_digest(candidate.encode("utf-8"), key.encode("utf-8")):
                found = i
        if found is not None:
            return found
        if settings.allow_unauthenticated and not header:
            return -1
        raise HTTPException(401, {"code": "unauthorized"}, headers={"WWW-Authenticate": "Bearer"})

    @app.get("/v1/models")
    async def models(request: Request):
        authenticate(request)
        return runtime.models()

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        started = time.perf_counter()
        admitted = False
        try:
            principal = authenticate(request)
            admitted = slots.acquire(blocking=False)
            if not admitted:
                raise HTTPException(529, {"code": "overloaded"}, headers={"Retry-After": "1"})
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > settings.max_body_bytes:
                    raise HTTPException(413, {"code": "request_body_too_large", "maximum": settings.max_body_bytes})
                chunks.append(chunk)
            try:
                parsed = SystemOneRequest.model_validate(strict_loads(b"".join(chunks)))
                encoding = await run_in_threadpool(runtime.encode, parsed)
            except ContextOverflow as exc:
                raise HTTPException(422, exc.detail) from exc
            except ValidationError as exc:
                # Never echo Pydantic's input values (may contain private content or credentials).
                detail = [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]}
                          for e in exc.errors(include_url=False, include_input=False)]
                raise HTTPException(422, detail) from exc
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise HTTPException(422, {"code": "invalid_json_or_request"}) from exc
            quotas.admit(principal, encoding.logical_tokens)
            response, compute = await run_in_threadpool(runtime.predict_encoded, encoding)
            hook.finish(elapsed_ms=(time.perf_counter() - started) * 1000, error=False,
                        input_tokens=response.usage.input_tokens, forward_tokens=response.usage.forward_tokens,
                        output_tokens=response.usage.output_tokens, questions=len(response.answers), prefix_reuses=compute["prefix_reuses"], execution=compute)
            # Preserve the native JSON response contract; execution counters are
            # content-free headers and also returned by predict_encoded().
            headers = {"X-Kev-Laya-Execution": compute["execution_version"],
                       "X-Kev-Laya-Prefix-Passes": str(compute["prefix_passes"]),
                       "X-Kev-Laya-Branch-Passes": str(compute["branch_passes"]),
                       "X-Kev-Laya-Batch-Sizes": ",".join(map(str, compute["effective_batch_sizes"])),
                       "X-Kev-Laya-Compute-Tokens": str(compute["compute_tokens"]),
                       "X-Kev-Laya-Padding-Tokens": str(compute["padding_tokens"])}
            return JSONResponse(content=response.model_dump(), headers=headers)
        except BatchBudgetExceeded as exc:
            hook.finish(elapsed_ms=(time.perf_counter() - started) * 1000, error=True)
            raise HTTPException(422, exc.detail) from exc
        except HTTPException:
            hook.finish(elapsed_ms=(time.perf_counter() - started) * 1000, error=True)
            raise
        except Exception:
            hook.finish(elapsed_ms=(time.perf_counter() - started) * 1000, error=True)
            return JSONResponse(status_code=500, content={"detail": {"code": "inference_failed"}})
        finally:
            if admitted:
                slots.release()
    return app
