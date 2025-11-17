#!/usr/bin/env python3
"""
concept_engine_loop.py - Infinite Concept Engine v0.1

Core idea:
- Load pxos_concept.md and pxos_backlog.yaml
- Pick the next TODO task
- Call an LLM with a stable prompt
- Apply the returned patches
- Mark tasks as done / add new tasks
- Repeat whenever you manually run this script
"""

import json
import yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
CONCEPT_FILE = ROOT / "pxos_concept.md"
BACKLOG_FILE = ROOT / "pxos_backlog.yaml"
PROMPT_FILE = ROOT / "prompts" / "infinite_engine_prompt.md"

# TODO: you wire this to OpenAI, Gemini, LM Studio, etc.
def call_model(messages):
    """Stub: replace with your actual LLM call."""
    raise NotImplementedError("Hook this up to your preferred model API / CLI.")

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def load_backlog():
    if not BACKLOG_FILE.exists():
        return {"version": 1, "tasks": [], "log": []}
    return yaml.safe_load(BACKLOG_FILE.read_text(encoding="utf-8"))

def save_backlog(data):
    BACKLOG_FILE.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

def pick_next_task(backlog):
    for task in backlog.get("tasks", []):
        if task.get("status") == "todo":
            return task
    return None

def apply_unified_diff(original_text: str, diff: str) -> str:
    """
    Minimal, dumb patch applier placeholder.
    For v0.1, you can just print the diff and use Aider/CLI to apply it.
    Later we can replace this with a real patch engine.
    """
    print("\n=== PATCH SUGGESTED ===")
    print(diff)
    print("=== END PATCH ===\n")
    # For now, return original text unchanged.
    return original_text

def main():
    concept = load_text(CONCEPT_FILE)
    backlog = load_backlog()
    prompt_base = load_text(PROMPT_FILE)

    task = pick_next_task(backlog)
    if not task:
        print("No TODO tasks found in pxos_backlog.yaml. Nothing to do.")
        return

    print(f"Selected task: {task['id']} - {task['title']}")

    # Build conversation for the model
    system_msg = {
        "role": "system",
        "content": prompt_base
    }

    user_msg = {
        "role": "user",
        "content": json.dumps({
            "concept": concept,
            "backlog": backlog,
            "task": task
        }, indent=2)
    }

    # Call the model
    try:
        response_text = call_model([system_msg, user_msg])
        result = json.loads(response_text)
    except NotImplementedError:
        print("call_model is not implemented. Skipping model call.")
        result = {}
    except json.JSONDecodeError:
        print("Model did not return valid JSON. Response was:")
        print(response_text)
        return

    print("\nMODEL THOUGHTS:")
    print(result.get("thoughts", ""))

    # Apply updates
    for upd in result.get("updates", []):
        file_path = ROOT / upd["file"]
        kind = upd.get("kind", "patch")
        content = upd.get("content", "")

        if kind == "patch":
            original = load_text(file_path)
            new_text = apply_unified_diff(original, content)
            save_text(file_path, new_text)
        elif kind == "replace":
            save_text(file_path, content)
        else:
            print(f"Unknown update kind '{kind}' for {file_path}")

    # Update backlog tasks
    completed_ids = set(result.get("completed_tasks", []))
    new_tasks = result.get("new_tasks", [])

    for task_entry in backlog.get("tasks", []):
        if task_entry.get("id") in completed_ids:
            task_entry["status"] = "done"

    if new_tasks:
        backlog["tasks"].extend(new_tasks)

    backlog.setdefault("log", []).append(
        f"{datetime.utcnow().isoformat()}Z: Engine processed task {task['id']}"
    )

    save_backlog(backlog)
    print("Backlog updated.")

if __name__ == "__main__":
    main()
