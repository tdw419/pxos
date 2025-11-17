#!/usr/bin/env python3
"""
Task Queue System for Pixel-LLM Development
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class Task:
    id: str
    name: str
    phase: int
    priority: int
    status: str
    agent: str
    dependencies: List[str]
    description: str
    created_at: float
    completed_at: Optional[float] = None
    result: Optional[str] = None

class TaskQueue:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.tasks: Dict[str, Task] = self._load()

    def _load(self) -> Dict[str, Task]:
        if not self.storage_path.exists():
            return {}
        with self.storage_path.open('r') as f:
            data = json.load(f)
        return {tid: Task(**tdata) for tid, tdata in data.items()}

    def _save(self):
        with self.storage_path.open('w') as f:
            json.dump({tid: asdict(task) for tid, task in self.tasks.items()}, f, indent=2)

    def add(self, name: str, phase: int, priority: int, agent: str, dependencies: List[str], description: str) -> Task:
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            phase=phase,
            priority=priority,
            status="PENDING",
            agent=agent,
            dependencies=dependencies,
            description=description,
            created_at=time.time()
        )
        self.tasks[task.id] = task
        self._save()
        return task

    def get_next(self) -> Optional[Task]:
        pending_tasks = [t for t in self.tasks.values() if t.status == "PENDING"]
        for task in sorted(pending_tasks, key=lambda t: (t.phase, -t.priority)):
            if all(self.tasks[dep_id].status == "COMPLETED" for dep_id in task.dependencies):
                return task
        return None

    def update_status(self, task_id: str, status: str, result: Optional[str] = None):
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if status in ["COMPLETED", "FAILED"]:
                self.tasks[task_id].completed_at = time.time()
                self.tasks[task_id].result = result
            self._save()

    def get_all(self) -> List[Task]:
        return list(self.tasks.values())
