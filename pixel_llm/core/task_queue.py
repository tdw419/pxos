#!/usr/bin/env python3
"""
Task Queue System for Pixel-LLM Development

Manages the coaching workflow where Gemini coaches local LLM
to build pixel-native AI infrastructure.

Key features:
- Priority-based task scheduling
- Task dependencies and phases
- Agent assignment (local_llm, gemini, human)
- Progress tracking and persistence
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


class TaskStatus(Enum):
    """Task lifecycle states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"  # Waiting for Gemini review
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class AgentType(Enum):
    """Available agents for task execution"""
    LOCAL_LLM = "local_llm"  # Local LLM (via llama.cpp)
    GEMINI = "gemini"        # Gemini API (for coaching/review)
    HUMAN = "human"          # Human intervention required
    AUTO = "auto"            # System decides


@dataclass
class Task:
    """Represents a single development task"""
    id: str
    title: str
    description: str
    action: str  # write_file, edit_file, review, test, etc.
    path: Optional[str] = None
    content: Optional[str] = None
    priority: int = 5  # 1-10, higher = more important
    status: TaskStatus = TaskStatus.PENDING
    preferred_agent: AgentType = AgentType.LOCAL_LLM
    phase: Optional[str] = None  # e.g., "1_storage", "2_inference"
    dependencies: List[str] = None  # Task IDs that must complete first
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    review_score: Optional[int] = None  # 1-10 from Gemini
    review_feedback: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        # Convert enums to their values if they're strings
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)
        if isinstance(self.preferred_agent, str):
            self.preferred_agent = AgentType(self.preferred_agent)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        data = asdict(self)
        data['status'] = self.status.value
        data['preferred_agent'] = self.preferred_agent.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create Task from dict"""
        # Handle enum conversions
        if 'status' in data:
            data['status'] = TaskStatus(data['status'])
        if 'preferred_agent' in data:
            data['preferred_agent'] = AgentType(data['preferred_agent'])
        return cls(**data)


class TaskQueue:
    """
    Manages the development task queue with priority scheduling,
    persistence, and agent coordination.
    """

    def __init__(self, storage_path: str = "pixel_llm/data/task_queue.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.tasks: Dict[str, Task] = {}
        self.load()

    def add_task(self, task_data: Dict) -> str:
        """
        Add a new task to the queue.

        Args:
            task_data: Dictionary with task fields

        Returns:
            Task ID (UUID)
        """
        task_id = str(uuid.uuid4())
        task_data['id'] = task_id

        # Set defaults
        if 'status' not in task_data:
            task_data['status'] = TaskStatus.PENDING
        if 'preferred_agent' not in task_data:
            task_data['preferred_agent'] = AgentType.LOCAL_LLM

        task = Task.from_dict(task_data)
        self.tasks[task_id] = task
        self.save()

        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)

    def get_next_task(self, agent: AgentType = AgentType.LOCAL_LLM) -> Optional[Task]:
        """
        Get the next highest-priority task for the specified agent.

        Considers:
        - Task status (PENDING only)
        - Dependencies (all must be COMPLETED)
        - Priority (higher first)
        - Agent preference

        Returns:
            Task object or None if no tasks available
        """
        eligible_tasks = []

        for task in self.tasks.values():
            # Must be pending
            if task.status != TaskStatus.PENDING:
                continue

            # Must be for this agent (or AUTO)
            if task.preferred_agent not in [agent, AgentType.AUTO]:
                continue

            # Check dependencies
            if not self._dependencies_met(task):
                continue

            # Must not exceed max attempts
            if task.attempts >= task.max_attempts:
                continue

            eligible_tasks.append(task)

        if not eligible_tasks:
            return None

        # Sort by priority (highest first), then creation time (oldest first)
        eligible_tasks.sort(
            key=lambda t: (-t.priority, t.created_at)
        )

        return eligible_tasks[0]

    def _dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are completed"""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    def start_task(self, task_id: str) -> bool:
        """Mark task as in progress"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now().isoformat()
        task.attempts += 1
        self.save()
        return True

    def complete_task(self, task_id: str, result: Dict = None) -> bool:
        """Mark task as completed with optional result data"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now().isoformat()

        if result:
            task.metadata['result'] = result
            if 'review_score' in result:
                task.review_score = result['review_score']
            if 'review_feedback' in result:
                task.review_feedback = result['review_feedback']

        self.save()
        return True

    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.FAILED
        task.metadata['error'] = error
        self.save()
        return True

    def send_to_review(self, task_id: str) -> bool:
        """Send task for Gemini review"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.REVIEW
        self.save()
        return True

    def get_phase_progress(self, phase: str) -> Dict[str, int]:
        """Get completion stats for a phase"""
        phase_tasks = [t for t in self.tasks.values() if t.phase == phase]

        if not phase_tasks:
            return {"total": 0, "completed": 0, "in_progress": 0, "pending": 0}

        return {
            "total": len(phase_tasks),
            "completed": sum(1 for t in phase_tasks if t.status == TaskStatus.COMPLETED),
            "in_progress": sum(1 for t in phase_tasks if t.status == TaskStatus.IN_PROGRESS),
            "pending": sum(1 for t in phase_tasks if t.status == TaskStatus.PENDING),
            "failed": sum(1 for t in phase_tasks if t.status == TaskStatus.FAILED),
        }

    def get_all_tasks(self, status: TaskStatus = None, phase: str = None) -> List[Task]:
        """Get all tasks, optionally filtered by status and/or phase"""
        tasks = list(self.tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]
        if phase:
            tasks = [t for t in tasks if t.phase == phase]

        return tasks

    def save(self):
        """Persist queue to disk"""
        data = {
            task_id: task.to_dict()
            for task_id, task in self.tasks.items()
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load queue from disk"""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self.tasks = {
                task_id: Task.from_dict(task_data)
                for task_id, task_data in data.items()
            }
        except Exception as e:
            print(f"Warning: Could not load task queue: {e}")
            self.tasks = {}

    def print_summary(self):
        """Print queue summary"""
        print("\n" + "="*60)
        print("PIXEL-LLM TASK QUEUE SUMMARY")
        print("="*60)

        # Overall stats
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS)
        pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)

        print(f"\nOverall: {completed}/{total} completed ({completed/total*100:.1f}%)")
        print(f"  In Progress: {in_progress}")
        print(f"  Pending: {pending}")

        # Phase breakdown
        phases = set(t.phase for t in self.tasks.values() if t.phase)
        if phases:
            print("\nPhase Progress:")
            for phase in sorted(phases):
                progress = self.get_phase_progress(phase)
                pct = progress['completed'] / progress['total'] * 100 if progress['total'] > 0 else 0
                print(f"  {phase}: {progress['completed']}/{progress['total']} ({pct:.0f}%)")

        print("="*60 + "\n")


# Global queue instance
_queue = None

def get_queue() -> TaskQueue:
    """Get global queue instance"""
    global _queue
    if _queue is None:
        _queue = TaskQueue()
    return _queue


# Convenience functions
def add_task(task_data: Dict) -> str:
    """Add task to queue"""
    return get_queue().add_task(task_data)


def get_next_task(agent: str = "local_llm") -> Optional[Task]:
    """Get next task for agent"""
    agent_type = AgentType(agent)
    return get_queue().get_next_task(agent_type)


def complete_task(task_id: str, result: Dict = None):
    """Mark task as completed"""
    return get_queue().complete_task(task_id, result)


def fail_task(task_id: str, error: str):
    """Mark task as failed"""
    return get_queue().fail_task(task_id, error)


# CLI for testing
if __name__ == "__main__":
    import sys

    queue = get_queue()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "add":
            # Add a test task
            task_id = add_task({
                "title": "Test task",
                "description": "This is a test task",
                "action": "write_file",
                "path": "test.py",
                "priority": 5
            })
            print(f"Added task: {task_id}")

        elif cmd == "next":
            task = get_next_task()
            if task:
                print(f"Next task: {task.title}")
                print(f"  ID: {task.id}")
                print(f"  Priority: {task.priority}")
                print(f"  Path: {task.path}")
            else:
                print("No tasks available")

        elif cmd == "list":
            tasks = queue.get_all_tasks()
            for task in sorted(tasks, key=lambda t: -t.priority):
                print(f"[{task.status.value:12s}] {task.title} (priority: {task.priority})")
    else:
        queue.print_summary()
