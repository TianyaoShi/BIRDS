from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence


class MockOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "BioLLMQualityMock/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._write_json({"object": "list", "data": [{"id": "mock-quality-model"}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in {"/v1/chat/completions", "/v1/completions"}:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        prompt = _prompt_from_payload(payload)
        model = str(payload.get("model") or "mock-quality-model")
        response_text = f"Mock response to: {prompt[:80]}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        chunk = _chunk_for_path(self.path, model=model, text=response_text)
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
        usage = {
            "choices": [],
            "usage": {
                "prompt_tokens": max(1, len(prompt.split())),
                "completion_tokens": max(1, len(response_text.split())),
            },
        }
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _prompt_from_payload(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = payload.get("messages")
    if isinstance(messages, list):
        parts = [
            str(message.get("content"))
            for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        ]
        return "\n".join(parts)
    return ""


def _chunk_for_path(path: str, *, model: str, text: str) -> dict[str, Any]:
    if path.rstrip("/").endswith("/chat/completions"):
        return {
            "id": "mock-chatcmpl",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    return {
        "id": "mock-cmpl",
        "object": "text_completion",
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": None}],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="output_quality_profiler.mock_openai_server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    server = ThreadingHTTPServer((args.host, args.port), MockOpenAIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

