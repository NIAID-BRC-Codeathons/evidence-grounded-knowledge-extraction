#!/usr/bin/env python3
"""A local OpenAI-compatible endpoint that fakes a generator.

Why this exists: `mango.cels.anl.gov` -- the host serving Qwen and Llama -- drops
TCP from this network on every port (the host answers ICMP, so it is a
source-address ACL, not a dead service). The hosted RAGStack backend cannot stand
in, because it owns its own server-side prompt and so cannot be given a batch of
passages. That leaves the parallel-batch path with no way to be exercised end to
end at all.

So this serves the two endpoints `LlmClient` uses and answers from the prompt it
was given. It is NOT a model and makes nothing up worth reading -- it reads the
column list and the `Source N` headers out of the prompt and emits one valid row
per source. That is exactly enough to exercise the parts that actually break:
per-batch citation numbering, the fan-out, the merge, and the table parser.

Deterministic by construction, so a test can assert on its output.

    python3 tools/stub_llm.py --port 8011
    litrag query -O "M. tuberculosis" -g katG -T mutation \
        --llm http://127.0.0.1:8011/v1 --llm-model stub

Standard library only: it has to run without the venv, and it must not become a
dependency of the package it is testing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_NAME = "stub-1"

# "Source 12 (Some title) [PMID: 3141592]:" -- the shape _source_block emits.
_SOURCE = re.compile(r"^Source (\d+)\b(.*)$", re.MULTILINE)
_COLUMNS = re.compile(r"exactly these columns:\n(.+)", re.MULTILINE)
# Every generation this process has served, so a test can ask what the fan-out
# actually sent rather than inferring it from the merged result.
CALLS: list = []


def _columns_from(prompt: str):
    match = _COLUMNS.search(prompt)
    if not match:
        return []
    return [c.strip() for c in match.group(1).split("\t") if c.strip()]


def _answer_for(prompt: str) -> str:
    """One TSV row per source in THIS prompt, citing that source's own marker.

    Citing the local marker is the whole point. Each parallel batch numbers its
    passages from 1, so a stub that cited a global index would hide precisely the
    bug the batching is at risk of introducing.
    """
    columns = _columns_from(prompt)
    if not columns:
        # Prose template, or a prompt shape we do not recognise. Say so rather
        # than emitting a table that would parse into silent nonsense.
        return "STUB: no column declaration found in prompt."

    markers = [int(m.group(1)) for m in _SOURCE.finditer(prompt)]
    lines = ["\t".join(columns)]
    for marker in markers:
        cells = []
        for column in columns:
            key = column.strip().lower()
            if key in {"reference", "references", "citation", "citations", "source"}:
                cells.append(f"[{marker}]")
            elif key in {"organism", "pathogen", "species"}:
                cells.append("Stub organism")
            elif key in {"gene name", "gene", "protein a", "protein"}:
                cells.append(f"stubGene{marker}")
            elif key == "protein b":
                cells.append(f"stubPartner{marker}")
            elif key == "mutation":
                # Distinct per source, so dedup has something real to merge and
                # something real to keep apart.
                cells.append(f"S{marker}T")
            else:
                cells.append(f"stub value {marker}")
        lines.append("\t".join(cells))
    return "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.path.rstrip("/").endswith("/models"):
            self._send({"data": [{"id": MODEL_NAME, "internal_id": MODEL_NAME}]})
        else:
            self._send({"error": f"no GET {self.path}"}, status=404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send({"error": "bad json"}, status=400)
            return

        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send({"error": f"no POST {self.path}"}, status=404)
            return

        messages = body.get("messages") or []
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        answer = _answer_for(prompt)
        CALLS.append({"at": time.time(), "prompt_chars": len(prompt),
                      "n_sources": len(_SOURCE.findall(prompt)),
                      "rows": answer.count("\n")})

        if self.server.delay:
            # Lets a caller prove the batches really do overlap in time: N
            # batches at D seconds each should finish in about D, not N*D.
            time.sleep(self.server.delay)

        self._send({
            "id": "stub", "model": MODEL_NAME,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": len(prompt) // 4,
                      "completion_tokens": len(answer) // 4},
        })

    def log_message(self, fmt, *args):
        if self.server.verbose:
            sys.stderr.write("stub_llm: " + (fmt % args) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep per call, to make concurrency visible.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    server.delay = args.delay
    server.verbose = not args.quiet
    sys.stderr.write(
        f"stub_llm on http://{args.host}:{args.port}/v1 "
        f"(model {MODEL_NAME}, delay {args.delay}s)\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write(f"\nstub_llm: {len(CALLS)} calls served\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
