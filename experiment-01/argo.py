#!/usr/bin/env python3
"""Call any Argo model from the command line, with the per-family parameter rules applied.

Standard library only, nothing to install. Requires the Argonne-auth network.

Usage:
    ./argo.py models                          # live model list with call IDs
    ./argo.py chat <model> "<prompt>"         # any of the chat models
    ./argo.py chat <model> "<prompt>" --max-tokens 8000 --system "You are a curator."
    ./argo.py embed <model> "<text>"          # v3large, v3small, ada002

The username comes from ARGO_USER, falling back to the codeathon helper script.
It is an identifier, not a password. Never pass the Argonne domain password here.

Parameter rules follow section 6 of the ANL-Argo-Quickstart README, measured on the
gateway on 2026-09-14:
- o-series (gpto1, gpto3, gpto3mini, gpto4mini): max_completion_tokens only, no temperature.
- Claude models on /v1/chat/completions: max_tokens is required and capped at 21000.
- gemini35flash: max_tokens of at least 2048, or the answer comes back empty.
- temperature is never sent, which every family accepts (claudesonnet5 and GPT-5.x
  reject any value other than 1).
"""

import argparse
import json
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request

BASE = "https://apps.inside.anl.gov/argoapi/v1"
HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".claude", "argo-user.sh")
O_SERIES = {"gpto1", "gpto3", "gpto3mini", "gpto4mini"}
CLAUDE_CAP = 21000


def username():
    user = os.environ.get("ARGO_USER")
    if not user and os.path.exists(HELPER):
        user = subprocess.run([HELPER], capture_output=True, text=True).stdout.strip()
    if not user:
        sys.exit("Set ARGO_USER to your Argonne collaborator username, e.g. ac.jdoe")
    return user


def tls_context():
    """Trust store for HTTPS.

    The python.org macOS build ships without usable root certificates, so Argo's public
    InCommon/USERTrust chain fails with CERTIFICATE_VERIFY_FAILED even though curl works.
    Prefer certifi when installed, then the macOS system roots, loaded in memory only.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    if sys.platform == "darwin":
        pem = subprocess.run(
            ["security", "find-certificate", "-a", "-p",
             "/System/Library/Keychains/SystemRootCertificates.keychain"],
            capture_output=True, text=True,
        ).stdout
        if "BEGIN CERTIFICATE" in pem:
            return ssl.create_default_context(cadata=pem)
    return ssl.create_default_context()


def call(path, payload=None, timeout=300):
    user = username()
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {user}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=tls_context()) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(f"HTTP {error.code}: {error.read().decode(errors='replace')[:500]}")
    except urllib.error.URLError as error:
        sys.exit(f"Cannot reach Argo ({error.reason}). Are you on the Argonne-auth network?")


def chat_payload(model, prompt, max_tokens, system):
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": messages}
    if model in O_SERIES:
        payload["max_completion_tokens"] = max_tokens
    else:
        limit = max_tokens
        if model.startswith("claude"):
            limit = min(limit, CLAUDE_CAP)
        if model == "gemini35flash":
            limit = max(limit, 2048)
        payload["max_tokens"] = limit
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models")
    chat = sub.add_parser("chat")
    chat.add_argument("model")
    chat.add_argument("prompt")
    chat.add_argument("--max-tokens", type=int, default=4096)
    chat.add_argument("--system")
    chat.add_argument("--json", action="store_true", help="print the raw response")
    embed = sub.add_parser("embed")
    embed.add_argument("model")
    embed.add_argument("text")
    args = parser.parse_args()

    if args.command == "models":
        for model in call("/models")["data"]:
            print(f"{model.get('internal_id'):<20} {model['id']:<26} {model.get('owned_by')}")
    elif args.command == "chat":
        prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
        data = call("/chat/completions", chat_payload(args.model, prompt, args.max_tokens, args.system))
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(data["choices"][0]["message"].get("content") or "")
    elif args.command == "embed":
        data = call("/embeddings", {"model": args.model, "input": args.text})
        vector = data["data"][0]["embedding"]
        print(f"{len(vector)} dimensions: {vector[:5]}")


if __name__ == "__main__":
    main()
