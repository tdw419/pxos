#!/usr/bin/env python3
"""
concept_engine_loop.py - Infinite Concept Engine v0.1

Core idea:
- Loads project state (concept, backlog)
- Picks the next TODO task
- Calls an LLM with the stable prompt
- Applies the returned patches
- Marks tasks as done / adds new tasks

This is the "autopilot" for pxOS development. Each run = one deliberate step forward.

Usage:
    python3 concept_engine_loop.py

Requirements:
    - pxos_concept.md (project state)
    - pxos_backlog.yaml (task queue)
    - prompts/infinite_engine_prompt.md (stable instructions)
    - LLM API configured in call_model()

For the concept to propagate infinitely, you need:
1. A concept with deep implications (pxOS has this)
2. A structured backlog (tasks → subtasks → sub-subtasks)
3. A stable prompt (maintains coherence)
4. An LLM that can follow instructions (Claude, GPT-4, Gemini)
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
import subprocess

# --- Configuration ---
ROOT = Path(__file__).parent
CONCEPT_FILE = ROOT / "pxos_concept.md"
BACKLOG_FILE = ROOT / "pxos_backlog.yaml"
PROMPT_FILE = ROOT / "prompts" / "infinite_engine_prompt.md"

# Model configuration (CUSTOMIZE THIS)
MODEL_PROVIDER = "claude"  # Options: "claude", "openai", "gemini", "local"
# ---------------------


def call_model(messages: list) -> str:
    """
    Call LLM API with messages and return response text.

    You must implement this based on your setup. Examples:

    For Claude API:
        import anthropic
        client = anthropic.Anthropic(api_key="...")
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            messages=messages
        )
        return response.content[0].text

    For OpenAI API:
        import openai
        client = openai.OpenAI(api_key="...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content

    For Gemini CLI:
        # Convert messages to prompt
        prompt = messages[1]["content"]
        result = subprocess.run(
            ["gemini", "prompt", prompt],
            capture_output=True, text=True
        )
        return result.stdout

    For Local LM Studio:
        import requests
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json={"model": "...", "messages": messages}
        )
        return response.json()["choices"][0]["message"]["content"]
    """

    # STUB: Replace with your actual implementation
    print("\n" + "=" * 60)
    print("CONCEPT ENGINE: Model call required")
    print("=" * 60)
    print(f"\nSystem prompt length: {len(messages[0]['content'])} chars")
    print(f"User context length: {len(messages[1]['content'])} chars")
    print("\nTo enable autonomous operation, implement call_model() with your LLM API.")
    print("For now, returning mock response.\n")

    return json.dumps({
        "thoughts": "Mock response. Implement call_model() to enable real operation.",
        "updates": [],
        "new_tasks": [],
        "completed_tasks": []
    })


def load_text(path: Path) -> str:
    """Load text file, return empty string if not found."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def save_text(path: Path, text: str):
    """Save text to file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_backlog():
    """Load backlog YAML, return default structure if not found."""
    if not BACKLOG_FILE.exists():
        return {"version": 1, "tasks": [], "log": []}
    return yaml.safe_load(BACKLOG_FILE.read_text(encoding="utf-8"))


def save_backlog(data):
    """Save backlog to YAML file."""
    BACKLOG_FILE.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="utf-8"
    )


def pick_next_task(backlog):
    """Pick the highest priority TODO task from backlog."""
    # Order by priority: critical > high > medium > low
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, None: 4}

    todo_tasks = [t for t in backlog.get("tasks", []) if t.get("status") == "todo"]
    if not todo_tasks:
        return None

    # Sort by priority, then by order in file
    todo_tasks.sort(key=lambda t: (priority_order.get(t.get("priority")), backlog["tasks"].index(t)))

    return todo_tasks[0]


def apply_unified_diff(original_text: str, diff: str) -> str:
    """
    Apply unified diff patch to original text.

    For v0.1, this is a stub that just prints the diff.
    In production, you'd use:
    - The 'patch' package (pip install patch)
    - subprocess call to 'git apply'
    - Manual patch parsing and application

    Returns original text unchanged (manual application required).
    """
    print("\n" + "=" * 60)
    print("PATCH SUGGESTED")
    print("=" * 60)
    print(diff)
    print("=" * 60)
    print("\nFor v0.1, please apply this patch manually.")
    print("Future versions will auto-apply patches.\n")

    return original_text


def main():
    """Main engine loop."""
    print("=" * 70)
    print(" pxOS INFINITE CONCEPT ENGINE v0.1")
    print("=" * 70)
    print()

    # Verify setup
    if not CONCEPT_FILE.exists():
        print(f"ERROR: {CONCEPT_FILE} not found.")
        print("Run: Initialize concept document first.")
        return

    if not BACKLOG_FILE.exists():
        print(f"ERROR: {BACKLOG_FILE} not found.")
        print("Run: Initialize backlog first.")
        return

    if not PROMPT_FILE.exists():
        print(f"ERROR: {PROMPT_FILE} not found.")
        print("Run: Initialize prompt first.")
        return

    # Load project state
    concept = load_text(CONCEPT_FILE)
    backlog = load_backlog()
    prompt_base = load_text(PROMPT_FILE)

    # Pick next task
    task = pick_next_task(backlog)

    if not task:
        print("✅ No TODO tasks found in backlog.")
        print("   All tasks complete, or add new tasks to continue development.\n")
        return

    print(f"📋 Selected Task: {task['id']} - {task['title']}")
    print(f"   Priority: {task.get('priority', 'unspecified')}")
    print(f"   Type: {task.get('type', 'unspecified')}")
    print()

    # Build conversation
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
    print("🤖 Calling LLM...")
    response_text = call_model([system_msg, user_msg])

    # Parse response
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"\n❌ ERROR: Model did not return valid JSON.")
        print(f"   Parse error: {e}")
        print(f"\n   Raw response:")
        print(response_text)
        return

    # Display thoughts
    print("\n" + "=" * 70)
    print(" MODEL REASONING")
    print("=" * 70)
    print(result.get("thoughts", "(no thoughts provided)"))
    print()

    # Apply updates
    updates = result.get("updates", [])
    if updates:
        print("=" * 70)
        print(f" APPLYING {len(updates)} UPDATE(S)")
        print("=" * 70)

        for i, upd in enumerate(updates, 1):
            file_path = ROOT / upd["file"]
            kind = upd.get("kind", "patch")
            content = upd.get("content", "")

            print(f"\n[{i}/{len(updates)}] {file_path}")
            print(f"         Kind: {kind}")

            if kind == "patch":
                original = load_text(file_path)
                new_text = apply_unified_diff(original, content)
                # TODO: Implement actual patch application
                # save_text(file_path, new_text)
            elif kind == "replace":
                save_text(file_path, content)
                print(f"         ✅ File replaced ({len(content)} bytes)")
            else:
                print(f"         ⚠️  Unknown update kind: {kind}")

    # Update backlog
    completed_ids = set(result.get("completed_tasks", []))
    new_tasks = result.get("new_tasks", [])

    if completed_ids:
        print("\n" + "=" * 70)
        print(f" MARKING {len(completed_ids)} TASK(S) COMPLETE")
        print("=" * 70)

        for task_entry in backlog.get("tasks", []):
            if task_entry.get("id") in completed_ids:
                task_entry["status"] = "done"
                print(f"   ✅ {task_entry['id']}: {task_entry['title']}")

    if new_tasks:
        print("\n" + "=" * 70)
        print(f" ADDING {len(new_tasks)} NEW TASK(S)")
        print("=" * 70)

        # Assign unique IDs
        existing_ids = [t.get("id", "") for t in backlog.get("tasks", [])]
        task_numbers = [int(tid[1:]) for tid in existing_ids if tid.startswith("T") and tid[1:].isdigit()]
        next_id_num = max(task_numbers, default=0) + 1

        for new_task in new_tasks:
            new_task["id"] = f"T{next_id_num:03d}"
            new_task["status"] = "todo"
            backlog["tasks"].append(new_task)
            print(f"   + {new_task['id']}: {new_task['title']}")
            next_id_num += 1

    # Update log
    timestamp = datetime.utcnow().isoformat() + "Z"
    log_entry = f"{timestamp}: Engine processed {task['id']}"

    if completed_ids:
        log_entry += f" (completed: {', '.join(completed_ids)})"
    if new_tasks:
        log_entry += f" (added {len(new_tasks)} tasks)"

    backlog.setdefault("log", []).append(log_entry)

    # Save backlog
    save_backlog(backlog)

    print("\n" + "=" * 70)
    print(" ENGINE CYCLE COMPLETE")
    print("=" * 70)
    print(f"\n✅ Backlog updated: {BACKLOG_FILE}")
    print(f"   Tasks remaining: {len([t for t in backlog['tasks'] if t['status'] == 'todo'])}")
    print()
    print("To continue development: Run this script again.")
    print("To change focus: Edit pxos_backlog.yaml")
    print()


if __name__ == "__main__":
    main()
