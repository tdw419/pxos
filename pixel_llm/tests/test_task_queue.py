#!/usr/bin/env python3
"""
Comprehensive unit tests for Task Queue system

Tests the development task queue including:
- Task creation and serialization
- Priority scheduling
- Dependency resolution
- Agent assignment
- Status transitions
- Persistence and recovery
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime

# Import TaskQueue components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from task_queue import Task, TaskQueue, TaskStatus, AgentType


class TestTask:
    """Test Task dataclass and serialization"""

    def test_task_creation_with_defaults(self):
        """Test task creates with sensible defaults"""
        task = Task(
            id="test-123",
            title="Test Task",
            description="A test task",
            action="write_file"
        )

        assert task.id == "test-123"
        assert task.status == TaskStatus.PENDING
        assert task.preferred_agent == AgentType.LOCAL_LLM
        assert task.priority == 5
        assert task.dependencies == []
        assert task.metadata == {}
        assert task.attempts == 0
        assert task.created_at is not None

    def test_task_to_dict(self):
        """Test task serialization to dict"""
        task = Task(
            id="test-123",
            title="Test Task",
            description="Test description",
            action="write_file",
            priority=7,
            phase="phase_1"
        )

        data = task.to_dict()

        assert data['id'] == "test-123"
        assert data['title'] == "Test Task"
        assert data['priority'] == 7
        assert data['status'] == "pending"  # Enum converted to value
        assert data['preferred_agent'] == "local_llm"

    def test_task_from_dict(self):
        """Test task deserialization from dict"""
        data = {
            'id': 'test-456',
            'title': 'Test Task',
            'description': 'Test',
            'action': 'edit_file',
            'status': 'completed',
            'preferred_agent': 'gemini',
            'priority': 8,
            'phase': 'phase_2'
        }

        task = Task.from_dict(data)

        assert task.id == 'test-456'
        assert task.status == TaskStatus.COMPLETED
        assert task.preferred_agent == AgentType.GEMINI
        assert task.priority == 8

    def test_task_round_trip_serialization(self):
        """Test task survives to_dict -> from_dict"""
        original = Task(
            id="test-789",
            title="Round trip test",
            description="Testing serialization",
            action="test",
            priority=9,
            phase="test_phase",
            dependencies=["dep-1", "dep-2"]
        )

        # Round trip
        data = original.to_dict()
        restored = Task.from_dict(data)

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.priority == original.priority
        assert restored.dependencies == original.dependencies


class TestTaskQueue:
    """Test TaskQueue operations"""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def queue(self, temp_storage):
        """Create TaskQueue instance"""
        storage_path = temp_storage / "task_queue.json"
        return TaskQueue(storage_path=str(storage_path))

    def test_queue_initialization(self, temp_storage):
        """Test queue initializes correctly"""
        storage_path = temp_storage / "task_queue.json"
        queue = TaskQueue(storage_path=str(storage_path))

        assert len(queue.tasks) == 0
        assert queue.storage_path.parent.exists()

    def test_add_task(self, queue):
        """Test adding a task"""
        task_id = queue.add_task({
            'title': 'Test Task',
            'description': 'Test description',
            'action': 'write_file',
            'priority': 7
        })

        assert task_id is not None
        assert task_id in queue.tasks
        assert queue.tasks[task_id].title == 'Test Task'
        assert queue.tasks[task_id].priority == 7

    def test_get_task(self, queue):
        """Test retrieving a task by ID"""
        task_id = queue.add_task({
            'title': 'Get Task Test',
            'description': 'Testing get_task',
            'action': 'test'
        })

        task = queue.get_task(task_id)

        assert task is not None
        assert task.id == task_id
        assert task.title == 'Get Task Test'

    def test_get_task_nonexistent(self, queue):
        """Test getting non-existent task returns None"""
        task = queue.get_task("fake-id-12345")
        assert task is None

    def test_start_task(self, queue):
        """Test starting a task"""
        task_id = queue.add_task({
            'title': 'Start Test',
            'description': 'Testing start',
            'action': 'test'
        })

        result = queue.start_task(task_id)
        task = queue.get_task(task_id)

        assert result is True
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None
        assert task.attempts == 1

    def test_complete_task(self, queue):
        """Test completing a task"""
        task_id = queue.add_task({
            'title': 'Complete Test',
            'description': 'Testing completion',
            'action': 'test'
        })

        queue.start_task(task_id)
        result = queue.complete_task(task_id)
        task = queue.get_task(task_id)

        assert result is True
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

    def test_complete_task_with_result(self, queue):
        """Test completing task with review results"""
        task_id = queue.add_task({
            'title': 'Complete with Result',
            'description': 'Testing',
            'action': 'test'
        })

        queue.complete_task(task_id, result={
            'review_score': 9,
            'review_feedback': 'Great work!'
        })

        task = queue.get_task(task_id)
        assert task.review_score == 9
        assert task.review_feedback == 'Great work!'

    def test_fail_task(self, queue):
        """Test failing a task"""
        task_id = queue.add_task({
            'title': 'Fail Test',
            'description': 'Testing failure',
            'action': 'test'
        })

        queue.start_task(task_id)
        result = queue.fail_task(task_id, "Test error message")
        task = queue.get_task(task_id)

        assert result is True
        assert task.status == TaskStatus.FAILED
        assert task.metadata['error'] == "Test error message"

    def test_send_to_review(self, queue):
        """Test sending task for review"""
        task_id = queue.add_task({
            'title': 'Review Test',
            'description': 'Testing review',
            'action': 'test'
        })

        result = queue.send_to_review(task_id)
        task = queue.get_task(task_id)

        assert result is True
        assert task.status == TaskStatus.REVIEW

    def test_get_next_task_simple(self, queue):
        """Test getting next task with no dependencies"""
        # Add tasks with different priorities
        queue.add_task({
            'title': 'Low Priority',
            'description': 'Test',
            'action': 'test',
            'priority': 3
        })

        high_id = queue.add_task({
            'title': 'High Priority',
            'description': 'Test',
            'action': 'test',
            'priority': 9
        })

        next_task = queue.get_next_task(AgentType.LOCAL_LLM)

        # Should get high priority task
        assert next_task is not None
        assert next_task.id == high_id
        assert next_task.priority == 9

    def test_get_next_task_with_dependencies(self, queue):
        """Test that tasks with unmet dependencies are skipped"""
        # Add dependency task
        dep_id = queue.add_task({
            'title': 'Dependency',
            'description': 'Must complete first',
            'action': 'test',
            'priority': 5
        })

        # Add dependent task
        queue.add_task({
            'title': 'Dependent Task',
            'description': 'Depends on first',
            'action': 'test',
            'priority': 10,  # Higher priority
            'dependencies': [dep_id]
        })

        # Next task should be dependency (not dependent)
        next_task = queue.get_next_task(AgentType.LOCAL_LLM)
        assert next_task.id == dep_id

    def test_get_next_task_dependencies_met(self, queue):
        """Test that tasks become available when dependencies complete"""
        # Add and complete dependency
        dep_id = queue.add_task({
            'title': 'Dependency',
            'description': 'Complete first',
            'action': 'test'
        })
        queue.start_task(dep_id)
        queue.complete_task(dep_id)

        # Add dependent task
        dep_task_id = queue.add_task({
            'title': 'Dependent',
            'description': 'Now available',
            'action': 'test',
            'dependencies': [dep_id]
        })

        next_task = queue.get_next_task(AgentType.LOCAL_LLM)
        assert next_task.id == dep_task_id

    def test_get_next_task_agent_filtering(self, queue):
        """Test that agent filtering works"""
        # Add LOCAL_LLM task
        queue.add_task({
            'title': 'Local Task',
            'description': 'For local LLM',
            'action': 'test',
            'preferred_agent': AgentType.LOCAL_LLM
        })

        # Add GEMINI task
        gemini_id = queue.add_task({
            'title': 'Gemini Task',
            'description': 'For Gemini',
            'action': 'test',
            'preferred_agent': AgentType.GEMINI
        })

        # Request Gemini task
        next_task = queue.get_next_task(AgentType.GEMINI)
        assert next_task.id == gemini_id

    def test_get_next_task_auto_agent(self, queue):
        """Test that AUTO tasks can be assigned to any agent"""
        auto_id = queue.add_task({
            'title': 'Auto Task',
            'description': 'Any agent can do this',
            'action': 'test',
            'preferred_agent': AgentType.AUTO
        })

        # Should be available to LOCAL_LLM
        next_task = queue.get_next_task(AgentType.LOCAL_LLM)
        assert next_task.id == auto_id

    def test_get_next_task_max_attempts(self, queue):
        """Test that tasks exceeding max attempts are skipped"""
        task_id = queue.add_task({
            'title': 'Max Attempts Test',
            'description': 'Will exceed attempts',
            'action': 'test',
            'max_attempts': 2
        })

        # Attempt twice
        queue.start_task(task_id)
        queue.fail_task(task_id, "First failure")

        task = queue.get_task(task_id)
        task.status = TaskStatus.PENDING  # Reset for next attempt

        queue.start_task(task_id)
        queue.fail_task(task_id, "Second failure")

        task.status = TaskStatus.PENDING  # Reset

        # Should not return this task (attempts=2, max=2)
        next_task = queue.get_next_task(AgentType.LOCAL_LLM)
        assert next_task is None or next_task.id != task_id

    def test_get_phase_progress(self, queue):
        """Test phase progress statistics"""
        # Add tasks in phase_1
        for i in range(3):
            queue.add_task({
                'title': f'Phase 1 Task {i}',
                'description': 'Test',
                'action': 'test',
                'phase': 'phase_1'
            })

        # Complete one
        tasks = queue.get_all_tasks(phase='phase_1')
        queue.start_task(tasks[0].id)
        queue.complete_task(tasks[0].id)

        progress = queue.get_phase_progress('phase_1')

        assert progress['total'] == 3
        assert progress['completed'] == 1
        assert progress['pending'] == 2

    def test_get_all_tasks_no_filter(self, queue):
        """Test getting all tasks without filters"""
        queue.add_task({'title': 'Task 1', 'description': 'Test', 'action': 'test'})
        queue.add_task({'title': 'Task 2', 'description': 'Test', 'action': 'test'})

        tasks = queue.get_all_tasks()
        assert len(tasks) == 2

    def test_get_all_tasks_status_filter(self, queue):
        """Test filtering tasks by status"""
        task1_id = queue.add_task({'title': 'Task 1', 'description': 'Test', 'action': 'test'})
        queue.add_task({'title': 'Task 2', 'description': 'Test', 'action': 'test'})

        queue.start_task(task1_id)
        queue.complete_task(task1_id)

        completed = queue.get_all_tasks(status=TaskStatus.COMPLETED)
        pending = queue.get_all_tasks(status=TaskStatus.PENDING)

        assert len(completed) == 1
        assert len(pending) == 1

    def test_get_all_tasks_phase_filter(self, queue):
        """Test filtering tasks by phase"""
        queue.add_task({'title': 'P1', 'description': 'Test', 'action': 'test', 'phase': 'phase_1'})
        queue.add_task({'title': 'P2', 'description': 'Test', 'action': 'test', 'phase': 'phase_2'})

        phase1_tasks = queue.get_all_tasks(phase='phase_1')
        assert len(phase1_tasks) == 1

    def test_persistence_save_load(self, temp_storage):
        """Test saving and loading queue"""
        storage_path = temp_storage / "queue.json"

        # Create queue and add tasks
        queue1 = TaskQueue(storage_path=str(storage_path))
        task_id = queue1.add_task({
            'title': 'Persist Test',
            'description': 'Testing persistence',
            'action': 'test',
            'priority': 8
        })

        # Create new queue instance (simulates restart)
        queue2 = TaskQueue(storage_path=str(storage_path))

        # Should load the task
        task = queue2.get_task(task_id)
        assert task is not None
        assert task.title == 'Persist Test'
        assert task.priority == 8

    def test_persistence_corrupted_file(self, temp_storage):
        """Test that corrupted queue file is handled gracefully"""
        storage_path = temp_storage / "queue.json"

        # Write corrupted JSON
        with open(storage_path, 'w') as f:
            f.write("{ invalid json ")

        # Should not crash
        queue = TaskQueue(storage_path=str(storage_path))
        assert len(queue.tasks) == 0


class TestTaskQueueEdgeCases:
    """Test edge cases and complex scenarios"""

    @pytest.fixture
    def queue(self):
        temp_dir = tempfile.mkdtemp()
        storage_path = Path(temp_dir) / "queue.json"
        q = TaskQueue(storage_path=str(storage_path))
        yield q
        shutil.rmtree(temp_dir)

    def test_empty_queue_operations(self, queue):
        """Test operations on empty queue"""
        assert queue.get_next_task(AgentType.LOCAL_LLM) is None
        assert queue.get_task("nonexistent") is None
        assert queue.get_all_tasks() == []

    def test_complex_dependency_chain(self, queue):
        """Test chain of dependencies (A -> B -> C)"""
        # Add tasks in reverse order
        task_c_id = queue.add_task({'title': 'C', 'description': 'Final', 'action': 'test'})

        task_b_id = queue.add_task({
            'title': 'B',
            'description': 'Middle',
            'action': 'test',
            'dependencies': [task_c_id]
        })

        queue.add_task({
            'title': 'A',
            'description': 'First dependent',
            'action': 'test',
            'dependencies': [task_b_id]
        })

        # Next task should be C (no dependencies)
        next_task = queue.get_next_task(AgentType.LOCAL_LLM)
        assert next_task.title == 'C'

    def test_missing_dependency(self, queue):
        """Test task with non-existent dependency"""
        queue.add_task({
            'title': 'Broken Dep',
            'description': 'Has missing dependency',
            'action': 'test',
            'dependencies': ['fake-dep-id']
        })

        # Should not return task with missing dependency
        next_task = queue.get_next_task(AgentType.LOCAL_LLM)
        assert next_task is None

    def test_priority_ordering_with_same_priority(self, queue):
        """Test that creation time is used as tiebreaker"""
        import time

        task1_id = queue.add_task({
            'title': 'First',
            'description': 'Created first',
            'action': 'test',
            'priority': 5
        })

        time.sleep(0.01)  # Ensure different timestamps

        queue.add_task({
            'title': 'Second',
            'description': 'Created second',
            'action': 'test',
            'priority': 5
        })

        next_task = queue.get_next_task(AgentType.LOCAL_LLM)
        # Should get older task
        assert next_task.id == task1_id

    def test_multiple_agents_different_tasks(self, queue):
        """Test multiple agents can work simultaneously"""
        local_id = queue.add_task({
            'title': 'Local Task',
            'description': 'For local',
            'action': 'test',
            'preferred_agent': AgentType.LOCAL_LLM
        })

        gemini_id = queue.add_task({
            'title': 'Gemini Task',
            'description': 'For Gemini',
            'action': 'test',
            'preferred_agent': AgentType.GEMINI
        })

        local_task = queue.get_next_task(AgentType.LOCAL_LLM)
        gemini_task = queue.get_next_task(AgentType.GEMINI)

        assert local_task.id == local_id
        assert gemini_task.id == gemini_id


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
