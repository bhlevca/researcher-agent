#!/usr/bin/env python3
"""Audit all installed Ollama models for tool-calling (function-calling) support.

Usage:
    python3 tools/check_ollama_tools.py [--base-url http://localhost:11434]

Exit code: 0 always.  Prints two sections:
  ✓  models that have {{.Tools}} in their chat template
  ✗  models that do not (planning fallback will kick in for these)

Models to keep even without tool support:
  - nomic-embed-text  — embedding model, used by CrewAI memory
  - deepseek-r1, qwq  — thinking/reasoning models, used for direct chat
"""

import argparse
import subprocess
import sys

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")


KNOWN_NON_CHAT = {"nomic-embed-text"}


def get_model_list(base_url: str) -> list[str]:
    r = requests.get(f"{base_url}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def probe(base_url: str, model: str) -> bool:
    """Return True if the model's template contains tool-calling support."""
    for payload in ({"name": model}, {"model": model}):
        try:
            r = requests.post(f"{base_url}/api/show", json=payload, timeout=10)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            tmpl = r.json().get("template", "")
            return (
                ".Tools" in tmpl
                or "{{- if .Tools }}" in tmpl
                or "<tools>" in tmpl.lower()
            )
        except Exception:
            pass
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    args = parser.parse_args()

    print(f"Probing Ollama at {args.base_url} …\n")
    try:
        models = get_model_list(args.base_url)
    except Exception as e:
        sys.exit(f"Cannot reach Ollama: {e}")

    has_tools, no_tools, embed_models = [], [], []

    for m in sorted(models):
        short = m.split(":")[0].split("/")[-1]
        if short in KNOWN_NON_CHAT:
            embed_models.append(m)
            continue
        if probe(args.base_url, m):
            has_tools.append(m)
        else:
            no_tools.append(m)

    col = 55
    print(f"{'=== HAS tool support':=<{col}}")
    for m in has_tools:
        print(f"  ✓  {m}")

    print()
    print(f"{'=== NO tool support':=<{col}}")
    for m in no_tools:
        print(f"  ✗  {m}")

    if embed_models:
        print()
        print(f"{'=== Embedding / special (keep regardless)':=<{col}}")
        for m in embed_models:
            print(f"  ~  {m}")

    print()
    if no_tools:
        print("To delete a model:  ollama rm <name>")
        print()
        print("Delete commands (review first!):")
        for m in no_tools:
            print(f"  ollama rm {m}")


if __name__ == "__main__":
    main()
