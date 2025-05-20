"""
Unit tests for PromptHistoryTracker and PromptHistoryEntry.
"""
import unittest
import uuid
from datetime import datetime, timedelta

from self_healing_agents.history import PromptHistoryEntry, PromptHistoryTracker
from self_healing_agents.error_types import AgentType

class TestPromptHistoryEntry(unittest.TestCase):
    def test_entry_creation_with_defaults(self):
        entry = PromptHistoryEntry(
            agent_type=AgentType.EXECUTOR,
            prompt_text="Test prompt",
            score=0.75,
            critic_report_summary={"status": "success"},
            task_id="task123",
            iteration_number=1,
            evo_iteration_number=0
        )
        self.assertIsNotNone(entry.prompt_id)
        self.assertTrue(isinstance(uuid.UUID(entry.prompt_id), uuid.UUID)) # Check if it's a valid UUID string
        self.assertIsNone(entry.parent_prompt_id)
        self.assertIsInstance(entry.timestamp, datetime)
        self.assertEqual(entry.agent_type, AgentType.EXECUTOR)
        self.assertEqual(entry.score, 0.75)

    def test_entry_creation_with_specific_id_and_parent(self):
        prompt_id = str(uuid.uuid4())
        parent_id = str(uuid.uuid4())
        ts = datetime.utcnow() - timedelta(seconds=10)
        entry = PromptHistoryEntry(
            prompt_id=prompt_id,
            parent_prompt_id=parent_id,
            agent_type=AgentType.PLANNER,
            prompt_text="Another prompt",
            score=0.5,
            critic_report_summary={"status": "failure", "error": "Logic error"},
            evolution_operator_used="enhance_decomposition",
            task_id="task456",
            iteration_number=2,
            evo_iteration_number=1,
            timestamp=ts
        )
        self.assertEqual(entry.prompt_id, prompt_id)
        self.assertEqual(entry.parent_prompt_id, parent_id)
        self.assertEqual(entry.evolution_operator_used, "enhance_decomposition")
        self.assertEqual(entry.timestamp, ts)

class TestPromptHistoryTracker(unittest.TestCase):
    def setUp(self):
        self.task_id = "tracker_task_001"
        self.tracker = PromptHistoryTracker(task_id=self.task_id)
        self.entry1_exec = PromptHistoryEntry(
            agent_type=AgentType.EXECUTOR,
            prompt_text="Exec prompt v1",
            score=0.5,
            critic_report_summary={"status": "failed", "reason": "syntax"},
            task_id=self.task_id,
            iteration_number=1,
            evo_iteration_number=0,
            timestamp=datetime.utcnow() - timedelta(minutes=5)
        )
        self.entry2_exec_child = PromptHistoryEntry(
            parent_prompt_id=self.entry1_exec.prompt_id,
            agent_type=AgentType.EXECUTOR,
            prompt_text="Exec prompt v2",
            score=0.8,
            critic_report_summary={"status": "success"},
            evolution_operator_used="error_specific_enhancement",
            task_id=self.task_id,
            iteration_number=1,
            evo_iteration_number=1,
            timestamp=datetime.utcnow() - timedelta(minutes=4)
        )
        self.entry3_plan = PromptHistoryEntry(
            agent_type=AgentType.PLANNER,
            prompt_text="Plan prompt v1",
            score=0.6,
            critic_report_summary={"status": "partial_success"},
            task_id=self.task_id,
            iteration_number=2,
            evo_iteration_number=0,
            timestamp=datetime.utcnow() - timedelta(minutes=3)
        )

    def test_initialization(self):
        self.assertEqual(self.tracker.task_id, self.task_id)
        self.assertEqual(len(self.tracker._history), 0)
        self.assertEqual(len(self.tracker._entries_by_agent[AgentType.EXECUTOR]), 0)
        self.assertEqual(len(self.tracker._entries_by_agent[AgentType.PLANNER]), 0)

    def test_add_entry_success(self):
        self.tracker.add_entry(self.entry1_exec)
        self.assertIn(self.entry1_exec.prompt_id, self.tracker._history)
        self.assertEqual(self.tracker._history[self.entry1_exec.prompt_id], self.entry1_exec)
        self.assertIn(self.entry1_exec.prompt_id, self.tracker._entries_by_agent[AgentType.EXECUTOR])

    def test_add_entry_duplicate_id_raises_value_error(self):
        self.tracker.add_entry(self.entry1_exec)
        with self.assertRaises(ValueError):
            self.tracker.add_entry(self.entry1_exec) # Adding same entry (same ID)

    def test_add_entry_mismatched_task_id_raises_value_error(self):
        wrong_task_entry = PromptHistoryEntry(
            agent_type=AgentType.EXECUTOR, prompt_text="Wrong task", score=0.1, 
            critic_report_summary={}, task_id="wrong_task", iteration_number=1, evo_iteration_number=0
        )
        with self.assertRaises(ValueError):
            self.tracker.add_entry(wrong_task_entry)

    def test_get_prompt_by_id(self):
        self.tracker.add_entry(self.entry1_exec)
        retrieved = self.tracker.get_prompt_by_id(self.entry1_exec.prompt_id)
        self.assertEqual(retrieved, self.entry1_exec)
        self.assertIsNone(self.tracker.get_prompt_by_id(str(uuid.uuid4()))) # Non-existent ID

    def test_get_prompts_for_agent(self):
        self.tracker.add_entry(self.entry1_exec)
        self.tracker.add_entry(self.entry2_exec_child)
        self.tracker.add_entry(self.entry3_plan)

        exec_prompts = self.tracker.get_prompts_for_agent(AgentType.EXECUTOR)
        self.assertEqual(len(exec_prompts), 2)
        self.assertIn(self.entry1_exec, exec_prompts)
        self.assertIn(self.entry2_exec_child, exec_prompts)
        self.assertEqual(exec_prompts[0].prompt_id, self.entry1_exec.prompt_id) # Sorted by timestamp
        self.assertEqual(exec_prompts[1].prompt_id, self.entry2_exec_child.prompt_id)

        plan_prompts = self.tracker.get_prompts_for_agent(AgentType.PLANNER)
        self.assertEqual(len(plan_prompts), 1)
        self.assertIn(self.entry3_plan, plan_prompts)

        # Test sorting false
        exec_prompts_unsorted = self.tracker.get_prompts_for_agent(AgentType.EXECUTOR, sort_by_timestamp=False)
        # Order might not be guaranteed without sort, but check presence
        self.assertIn(self.entry1_exec, exec_prompts_unsorted)
        self.assertIn(self.entry2_exec_child, exec_prompts_unsorted)

    def test_get_lineage(self):
        self.tracker.add_entry(self.entry1_exec)
        self.tracker.add_entry(self.entry2_exec_child) # Child of entry1_exec
        self.tracker.add_entry(self.entry3_plan) # Unrelated lineage

        # Lineage of child should include child and parent
        lineage_child = self.tracker.get_lineage(self.entry2_exec_child.prompt_id)
        self.assertEqual(len(lineage_child), 2)
        self.assertEqual(lineage_child[0].prompt_id, self.entry2_exec_child.prompt_id)
        self.assertEqual(lineage_child[1].prompt_id, self.entry1_exec.prompt_id)

        # Lineage of parent should only include parent
        lineage_parent = self.tracker.get_lineage(self.entry1_exec.prompt_id)
        self.assertEqual(len(lineage_parent), 1)
        self.assertEqual(lineage_parent[0].prompt_id, self.entry1_exec.prompt_id)

        # Lineage of unrelated entry
        lineage_plan = self.tracker.get_lineage(self.entry3_plan.prompt_id)
        self.assertEqual(len(lineage_plan), 1)
        self.assertEqual(lineage_plan[0].prompt_id, self.entry3_plan.prompt_id)

        # Lineage of non-existent ID
        self.assertEqual(len(self.tracker.get_lineage(str(uuid.uuid4()))), 0)

    def test_get_full_history(self):
        self.tracker.add_entry(self.entry1_exec)
        self.tracker.add_entry(self.entry2_exec_child)
        self.tracker.add_entry(self.entry3_plan)

        full_history = self.tracker.get_full_history()
        self.assertEqual(len(full_history), 3)
        self.assertIn(self.entry1_exec, full_history)
        self.assertIn(self.entry2_exec_child, full_history)
        self.assertIn(self.entry3_plan, full_history)
        # Check sorted order by timestamp
        self.assertEqual(full_history[0].prompt_id, self.entry1_exec.prompt_id)
        self.assertEqual(full_history[1].prompt_id, self.entry2_exec_child.prompt_id)
        self.assertEqual(full_history[2].prompt_id, self.entry3_plan.prompt_id)

        full_history_unsorted = self.tracker.get_full_history(sort_by_timestamp=False)
        self.assertEqual(len(full_history_unsorted), 3)
        self.assertIn(self.entry1_exec, full_history_unsorted)
        self.assertIn(self.entry2_exec_child, full_history_unsorted)
        self.assertIn(self.entry3_plan, full_history_unsorted)

if __name__ == '__main__':
    unittest.main() 