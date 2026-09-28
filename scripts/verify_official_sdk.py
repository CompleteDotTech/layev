"""Official TypeSafe SDK gate; synthetic self-test never contacts an external API."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

from typesafe_sdk import (
    Choice, Noul, Score, TypeSafeAuthenticationError, TypeSafeClient,
    TypeSafeUnprocessableEntityError,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def probe(base_url: str, model: str, api_key: str) -> dict:
    parsed = urlsplit(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username):
        raise ValueError("SDK acceptance requires an explicit loopback HTTP endpoint")
    if not model.startswith("kev-laya-") or not api_key:
        raise ValueError("explicit Layev model and API key are required")

    options = {"red": None, "blue": None, "green": None}
    levels = ["low", "medium", "high"]
    with TypeSafeClient(base_url=base_url, model=model, api_key=api_key, timeout=5.0) as client:
        response = client.system_one(
            state={"color": "red", "level": 1, "context": ["structured", "state"]},
            questions={
                "color": Choice(instructions={"task": "choose the color"}, criteria=options),
                "flag": Noul(instructions="Is the color red?"),
                "level": Score(instructions="Rate urgency", criteria=levels),
            },
        )
    require(response.model.startswith("kev-laya-"), "resolved Layev model identity missing")
    require(set(response.answers) == {"color", "flag", "level"}, "answer coverage differs from request")
    choice, noul, score = response.choices["color"], response.nouls["flag"], response.scores["level"]
    require(choice.choice in options, "choice is outside dynamic options")
    require(set(choice.probabilities) == set(options), "choice probability keys differ")
    require(abs(sum(choice.probabilities.values()) - 1) < 1e-6, "choice probabilities do not sum to one")
    require(0 <= choice.confidence <= 1 and 0 <= noul.noul <= 1, "invalid choice/noul confidence")
    require(0 <= score.score <= len(levels) - 1 and 0 <= score.confidence <= 1, "invalid score")
    require(set(score.probabilities) == set(range(len(levels))), "score probability keys differ")
    require(set(score.legend) == set(range(len(levels))), "score legend differs")
    require(abs(sum(score.probabilities.values()) - 1) < 1e-6, "score probabilities do not sum to one")
    require(response.usage.input_tokens is not None and response.usage.input_tokens > 0, "input usage missing")
    require(response.usage.output_tokens is not None and response.usage.output_tokens >= 0, "output usage missing")
    return {
        "requested_model": model,
        "resolved_model": response.model,
        "answer_types": {key: answer.type for key, answer in response.answers.items()},
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "choice_options": sorted(choice.probabilities),
        "score_legend": score.legend,
    }


def probe_errors(base_url: str, model: str, api_key: str) -> dict:
    try:
        probe(base_url, model, api_key + "-incorrect")
    except TypeSafeAuthenticationError:
        pass
    else:
        raise ValueError("official SDK did not reject an invalid key as HTTP 401")
    try:
        probe(base_url, "kev-laya-missing", api_key)
    except TypeSafeUnprocessableEntityError:
        pass
    else:
        raise ValueError("official SDK did not reject an unknown model as HTTP 422")
    return {"invalid_key": "typed_401", "unknown_model": "typed_422"}


@contextmanager
def synthetic_server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/systemone":
                self.send_error(404)
                return
            size = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(size))
            if self.headers.get("Authorization") != "Bearer local-test":
                self.send_response(401)
                self.end_headers()
                return
            if request["model"] != "kev-laya-test":
                self.send_response(422)
                self.end_headers()
                return
            require(set(request["questions"]) == {"color", "flag", "level"}, "question ids differ")
            require(set(request["questions"]["color"]["criteria"]) == {"red", "blue", "green"}, "options differ")
            body = json.dumps({
                "model": "kev-laya-test",
                "answers": {
                    "color": {"type": "choice", "choice": "red", "confidence": 0.8,
                              "probabilities": {"red": 0.8, "blue": 0.1, "green": 0.1}},
                    "flag": {"type": "noul", "noul": 0.75},
                    "level": {"type": "score", "score": 1.25, "confidence": 0.7,
                              "probabilities": {"0": 0.1, "1": 0.55, "2": 0.35},
                              "legend": {"0": "low", "1": "medium", "2": "high"}},
                },
                "usage": {"input_tokens": 12, "output_tokens": 5},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["self-test", "installed-service"])
    parser.add_argument("--out", type=Path, help="new private JSON receipt; never overwritten")
    args = parser.parse_args()
    sdk_version = version("typesafe-sdk")
    if sdk_version != "0.7.2":
        print(f"BLOCKED: tested official typesafe-sdk 0.7.2 required; found {sdk_version}")
        return 2
    if args.mode == "self-test":
        with synthetic_server() as base:
            probe(base, "kev-laya-test", "local-test")
            probe_errors(base, "kev-laya-test", "local-test")
        print("official SDK synthetic loopback contract passed; trained service unverified")
        return 0
    base = os.environ.get("KEV_LAYA_BASE_URL")
    model = os.environ.get("KEV_LAYA_MODEL")
    key = os.environ.get("KEV_LAYA_TEST_API_KEY")
    if not base or not model or not key:
        print("BLOCKED: explicit installed-service URL, model and API key required")
        return 2
    if args.out is None or not args.out.parent.is_dir() or args.out.exists():
        print("BLOCKED: --out must be a new path in an existing private directory")
        return 2
    result = probe(base, model, key)
    result["errors"] = probe_errors(base, model, key)
    result.update({
        "gate": "official-typesafe-sdk-installed-service-v1",
        "sdk_version": sdk_version,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "endpoint": "loopback",
        "trained_checkpoint_provenance": "verify_separately",
        "quality": "unmeasured",
    })
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("official SDK installed-service interface probe passed; verify checkpoint provenance separately")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
