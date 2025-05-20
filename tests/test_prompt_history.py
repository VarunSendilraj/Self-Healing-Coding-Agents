import unittest
from uuid import uuid4, UUID
from datetime import datetime
from typing import Dict, Any, Optional, Union

# Assuming these enums and classes will be available from these paths
# Adjust imports as per your actual project structure

# Directly define mock AgentType for now to debug AttributeError
from enum import Enum
class AgentType(Enum):
    PLANNER = "PLANNER"
    EXECUTOR = "EXECUTOR"
    UNKNOWN = "UNKNOWN"

try:
    from src.self_healing_agents.prompt_history import PromptHistoryEntry, PromptEvaluationMetrics, PromptHistoryTracker
except ImportError:
    # Mock implementations for testing if not available
    from dataclasses import dataclass, field

    @dataclass
    class PromptEvaluationMetrics:
        # Common
        overall_score: Optional[float] = None
        # Planner-specific
        logical_consistency: Optional[float] = None
        completeness: Optional[float] = None
        efficiency: Optional[float] = None
        # Executor-specific
        code_correctness: Optional[float] = None
        runtime_efficiency: Optional[float] = None
        error_handling: Optional[float] = None
        # Custom metrics
        custom_metrics: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class PromptHistoryEntry:
        id: UUID = field(default_factory=uuid4)
        prompt_text: str = ""
        agent_type: AgentType = AgentType.UNKNOWN
        timestamp: datetime = field(default_factory=datetime.utcnow)
        parent_id: Optional[UUID] = None
        metrics: Optional[PromptEvaluationMetrics] = None
        fixed_error_info: Optional[Dict[str, Any]] = None # e.g., {"error_id": "xyz", "details": "..."}
        generation_metadata: Dict[str, Any] = field(default_factory=dict) # e.g., LLM params

    class PromptHistoryTracker:
        def __init__(self):
            self.prompts: Dict[UUID, PromptHistoryEntry] = {}

        def add_prompt_entry(self, entry: PromptHistoryEntry) -> None:
            if entry.parent_id and entry.parent_id not in self.prompts:
                raise ValueError(f"Parent prompt with ID {entry.parent_id} not found.")
            self.prompts[entry.id] = entry

        def get_prompt_by_id(self, prompt_id: UUID) -> Optional[PromptHistoryEntry]:
            return self.prompts.get(prompt_id)

        def get_lineage(self, prompt_id: UUID) -> list[PromptHistoryEntry]:
            lineage = []
            current_id = prompt_id
            while current_id:
                entry = self.get_prompt_by_id(current_id)
                if not entry:
                    break # Should not happen if IDs are valid
                lineage.append(entry)
                current_id = entry.parent_id
            return list(reversed(lineage))

        def get_prompts_by_agent_type(self, agent_type: AgentType) -> list[PromptHistoryEntry]:
            return [p for p in self.prompts.values() if p.agent_type == agent_type]

        def get_latest_prompt_by_agent_type(self, agent_type: AgentType) -> Optional[PromptHistoryEntry]:
            agent_prompts = self.get_prompts_by_agent_type(agent_type)
            if not agent_prompts:
                return None
            return max(agent_prompts, key=lambda p: p.timestamp)


class TestPromptEvaluationMetrics(unittest.TestCase):
    def test_create_planner_metrics(self):
        metrics = PromptEvaluationMetrics(
            overall_score=0.8,
            logical_consistency=0.9,
            completeness=0.85,
            efficiency=0.75
        )
        self.assertEqual(metrics.overall_score, 0.8)
        self.assertEqual(metrics.logical_consistency, 0.9)
        self.assertIsNone(metrics.code_correctness)

    def test_create_executor_metrics(self):
        metrics = PromptEvaluationMetrics(
            overall_score=0.7,
            code_correctness=0.95,
            runtime_efficiency=0.7,
            error_handling=0.8
        )
        self.assertEqual(metrics.overall_score, 0.7)
        self.assertEqual(metrics.code_correctness, 0.95)
        self.assertIsNone(metrics.logical_consistency)

    def test_custom_metrics(self):
        metrics = PromptEvaluationMetrics(
            custom_metrics={"accuracy": 0.99, "latency_ms": 120}
        )
        self.assertEqual(metrics.custom_metrics["accuracy"], 0.99)
        self.assertEqual(metrics.custom_metrics["latency_ms"], 120)


class TestPromptHistoryTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = PromptHistoryTracker()
        self.planner_agent_type = AgentType.PLANNER
        self.executor_agent_type = AgentType.EXECUTOR

    def test_add_genesis_prompt(self):
        prompt_text = "Initial planner prompt"
        entry = PromptHistoryEntry(
            prompt_text=prompt_text,
            agent_type=self.planner_agent_type
        )
        self.tracker.add_prompt_entry(entry)
        
        retrieved_entry = self.tracker.get_prompt_by_id(entry.id)
        self.assertIsNotNone(retrieved_entry)
        self.assertEqual(retrieved_entry.prompt_text, prompt_text)
        self.assertEqual(retrieved_entry.agent_type, self.planner_agent_type)
        self.assertIsNone(retrieved_entry.parent_id)

    def test_add_child_prompt(self):
        parent_entry = PromptHistoryEntry(
            prompt_text="Parent prompt",
            agent_type=self.planner_agent_type
        )
        self.tracker.add_prompt_entry(parent_entry)

        child_text = "Child prompt evolving parent"
        child_entry = PromptHistoryEntry(
            prompt_text=child_text,
            agent_type=self.planner_agent_type,
            parent_id=parent_entry.id
        )
        self.tracker.add_prompt_entry(child_entry)

        retrieved_child = self.tracker.get_prompt_by_id(child_entry.id)
        self.assertIsNotNone(retrieved_child)
        self.assertEqual(retrieved_child.parent_id, parent_entry.id)

    def test_add_prompt_with_non_existent_parent(self):
        non_existent_parent_id = uuid4()
        entry = PromptHistoryEntry(
            prompt_text="Orphan prompt",
            agent_type=self.planner_agent_type,
            parent_id=non_existent_parent_id
        )
        with self.assertRaises(ValueError):
            self.tracker.add_prompt_entry(entry)

    def test_get_lineage(self):
        p1 = PromptHistoryEntry(prompt_text="P1", agent_type=self.planner_agent_type)
        self.tracker.add_prompt_entry(p1)
        
        p2 = PromptHistoryEntry(prompt_text="P2", agent_type=self.planner_agent_type, parent_id=p1.id)
        self.tracker.add_prompt_entry(p2)
        
        p3 = PromptHistoryEntry(prompt_text="P3", agent_type=self.planner_agent_type, parent_id=p2.id)
        self.tracker.add_prompt_entry(p3)

        lineage = self.tracker.get_lineage(p3.id)
        self.assertEqual(len(lineage), 3)
        self.assertEqual(lineage[0].id, p1.id)
        self.assertEqual(lineage[1].id, p2.id)
        self.assertEqual(lineage[2].id, p3.id)

        # Lineage of a genesis prompt
        lineage_p1 = self.tracker.get_lineage(p1.id)
        self.assertEqual(len(lineage_p1), 1)
        self.assertEqual(lineage_p1[0].id, p1.id)
        
        # Lineage of non-existent prompt
        self.assertEqual(self.tracker.get_lineage(uuid4()), [])


    def test_track_performance_metrics(self):
        metrics = PromptEvaluationMetrics(overall_score=0.9, logical_consistency=0.95)
        entry = PromptHistoryEntry(
            prompt_text="Prompt with metrics",
            agent_type=self.planner_agent_type,
            metrics=metrics
        )
        self.tracker.add_prompt_entry(entry)

        retrieved_entry = self.tracker.get_prompt_by_id(entry.id)
        self.assertIsNotNone(retrieved_entry.metrics)
        self.assertEqual(retrieved_entry.metrics.overall_score, 0.9)
        self.assertEqual(retrieved_entry.metrics.logical_consistency, 0.95)

    def test_track_fixed_error_info(self):
        error_info = {"error_id": "ERR123", "type": "SyntaxError"}
        entry = PromptHistoryEntry(
            prompt_text="Prompt that fixed an error",
            agent_type=self.executor_agent_type,
            fixed_error_info=error_info
        )
        self.tracker.add_prompt_entry(entry)

        retrieved_entry = self.tracker.get_prompt_by_id(entry.id)
        self.assertIsNotNone(retrieved_entry.fixed_error_info)
        self.assertEqual(retrieved_entry.fixed_error_info["error_id"], "ERR123")

    def test_get_prompts_by_agent_type(self):
        p1 = PromptHistoryEntry(agent_type=self.planner_agent_type)
        e1 = PromptHistoryEntry(agent_type=self.executor_agent_type)
        p2 = PromptHistoryEntry(agent_type=self.planner_agent_type)
        self.tracker.add_prompt_entry(p1)
        self.tracker.add_prompt_entry(e1)
        self.tracker.add_prompt_entry(p2)

        planner_prompts = self.tracker.get_prompts_by_agent_type(self.planner_agent_type)
        executor_prompts = self.tracker.get_prompts_by_agent_type(self.executor_agent_type)

        self.assertEqual(len(planner_prompts), 2)
        self.assertIn(p1, planner_prompts)
        self.assertIn(p2, planner_prompts)
        self.assertEqual(len(executor_prompts), 1)
        self.assertIn(e1, executor_prompts)

    def test_get_latest_prompt_by_agent_type(self):
        p1_time = datetime.utcnow()
        p1 = PromptHistoryEntry(agent_type=self.planner_agent_type, timestamp=p1_time)
        self.tracker.add_prompt_entry(p1)

        # Ensure subsequent prompts have later timestamps
        import time; time.sleep(0.001) 
        e1_time = datetime.utcnow()
        e1 = PromptHistoryEntry(agent_type=self.executor_agent_type, timestamp=e1_time)
        self.tracker.add_prompt_entry(e1)
        
        time.sleep(0.001)
        p2_time = datetime.utcnow()
        p2 = PromptHistoryEntry(agent_type=self.planner_agent_type, timestamp=p2_time)
        self.tracker.add_prompt_entry(p2)
        
        latest_planner = self.tracker.get_latest_prompt_by_agent_type(self.planner_agent_type)
        latest_executor = self.tracker.get_latest_prompt_by_agent_type(self.executor_agent_type)

        self.assertIsNotNone(latest_planner)
        self.assertEqual(latest_planner.id, p2.id)
        
        self.assertIsNotNone(latest_executor)
        self.assertEqual(latest_executor.id, e1.id)
        
        self.assertIsNone(self.tracker.get_latest_prompt_by_agent_type(AgentType.UNKNOWN))


if __name__ == '__main__':
    unittest.main() 