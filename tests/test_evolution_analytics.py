import unittest
from unittest import mock # Added for mocking plot functions
from uuid import uuid4, UUID
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter
import pandas as pd # Added for heatmap data structuring

# Mock plotting libraries if not available, or for headless environments
try:
    import matplotlib.pyplot as plt
    import seaborn
    import networkx as nx
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    # Create dummy decorators or mocks if needed for plt.show() or similar during testing without display
    # For savefig, the mock will handle it.
    class MockPlt:
        def savefig(self, *args, **kwargs): pass
        def close(self, *args, **kwargs): pass
        def figure(self, *args, **kwargs): return self # Allow chaining
        def subplots(self, *args, **kwargs): 
            mock_ax = MockAx()
            return self, mock_ax # Return fig, ax
        def title(self, *args, **kwargs): pass
        def xlabel(self, *args, **kwargs): pass
        def ylabel(self, *args, **kwargs): pass
        def gca(self, *args, **kwargs): return MockAx() # Return mock axis
        def tight_layout(self, *args, **kwargs): pass # Added for heatmap

    class MockAx:
        def clear(self, *args, **kwargs): pass
        def set_xticks(self, *args, **kwargs): pass
        def set_yticks(self, *args, **kwargs): pass
        def set_xticklabels(self, *args, **kwargs): pass
        def set_yticklabels(self, *args, **kwargs): pass

    plt = MockPlt()
    seaborn = mock.Mock() # Seaborn can be a simple mock for heatmap call
    nx = mock.Mock()      # NetworkX can be a simple mock for draw calls

# Directly define mock Enums for now to debug AttributeError
from enum import Enum
class AgentType(Enum):
    PLANNER = "PLANNER"
    EXECUTOR = "EXECUTOR"
    UNKNOWN = "UNKNOWN"

class ErrorType(Enum):
    PLANNING_ERROR = "PLANNING_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    TYPE_ERROR = "TYPE_ERROR"
    LOGIC_ERROR = "LOGIC_ERROR"
    AMBIGUOUS_ERROR = "AMBIGUOUS_ERROR"
    NONE = "NONE"

# Assuming these enums and classes will be available from these paths
# Adjust imports as per your actual project structure
# try:
#     from src.self_healing_agents.error_types import AgentType, ErrorType # Assuming ErrorType enum exists
# except ImportError:
#     from enum import Enum
#     class AgentType(Enum):
#         PLANNER = "PLANNER"
#         EXECUTOR = "EXECUTOR"
#         UNKNOWN = "UNKNOWN"
# 
#     class ErrorType(Enum): # Mock ErrorType
#         PLANNING_ERROR = "PLANNING_ERROR"
#         EXECUTION_ERROR = "EXECUTION_ERROR"
#         SYNTAX_ERROR = "SYNTAX_ERROR"
#         TYPE_ERROR = "TYPE_ERROR"
#         LOGIC_ERROR = "LOGIC_ERROR"
#         AMBIGUOUS_ERROR = "AMBIGUOUS_ERROR"
#         NONE = "NONE"


try:
    # If PromptHistoryEntry and PromptEvaluationMetrics are in a shared location:
    from src.self_healing_agents.prompt_history import PromptHistoryEntry, PromptEvaluationMetrics
except ImportError:
    # Mock implementations from test_prompt_history.py (ensure consistency or centralize mocks)
    from dataclasses import dataclass, field

    @dataclass
    class PromptEvaluationMetrics:
        overall_score: Optional[float] = None
        logical_consistency: Optional[float] = None
        completeness: Optional[float] = None
        efficiency: Optional[float] = None
        code_correctness: Optional[float] = None
        runtime_efficiency: Optional[float] = None
        error_handling: Optional[float] = None
        custom_metrics: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class PromptHistoryEntry:
        id: UUID = field(default_factory=uuid4)
        prompt_text: str = ""
        agent_type: AgentType = AgentType.UNKNOWN
        timestamp: datetime = field(default_factory=datetime.utcnow)
        parent_id: Optional[UUID] = None
        metrics: Optional[PromptEvaluationMetrics] = None
        # fixed_error_info: Optional[Dict[str, Any]] = None
        # Example: {"error_id": "xyz", "original_error_type": ErrorType.SYNTAX_ERROR, "details": "..."}
        fixed_error_info: Optional[Dict[str, Any]] = None
         # E.g., {"operator_name": "add_examples", "llm_params": {...}}
        generation_metadata: Dict[str, Any] = field(default_factory=dict)
        succeeded: Optional[bool] = None # True if this prompt led to a successful outcome/fix


class EvolutionAnalytics:
    def __init__(self, history: List[PromptHistoryEntry]):
        self.history = sorted(history, key=lambda x: x.timestamp)

    def get_modification_effectiveness(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Calculates effectiveness of prompt modifications.
        Returns: Dict[modification_strategy, {"success_rate": float, "total_applications": int}]
        """
        strategy_counts: Dict[str, Dict[str, int]] = {} # strategy -> {"success": N, "total": M}

        for entry in self.history:
            strategy = entry.generation_metadata.get("operator_name", "unknown_operator")
            if strategy not in strategy_counts:
                strategy_counts[strategy] = {"success": 0, "total": 0}
            
            strategy_counts[strategy]["total"] += 1
            if entry.succeeded:
                strategy_counts[strategy]["success"] += 1
        
        effectiveness = {}
        for strategy, counts in strategy_counts.items():
            effectiveness[strategy] = {
                "success_rate": counts["success"] / counts["total"] if counts["total"] > 0 else 0.0,
                "total_applications": counts["total"],
                "successful_applications": counts["success"]
            }
        return effectiveness

    def get_error_fix_correlation(self) -> Dict[ErrorType, Dict[str, int]]:
        """
        Correlates error types with successful fix strategies.
        Returns: Dict[error_type, Dict[successful_strategy, count]]
        """
        correlations: Dict[ErrorType, Counter[str]] = {}

        for entry in self.history:
            if entry.succeeded and entry.fixed_error_info:
                error_type_str = entry.fixed_error_info.get("original_error_type")
                # Ensure error_type_str is converted to ErrorType enum if it's a string
                try:
                    error_type = ErrorType(error_type_str) if isinstance(error_type_str, str) else error_type_str
                    if not isinstance(error_type, ErrorType): # Fallback if conversion failed
                         error_type = ErrorType.AMBIGUOUS_ERROR
                except ValueError:
                    error_type = ErrorType.AMBIGUOUS_ERROR


                strategy = entry.generation_metadata.get("operator_name", "unknown_operator")
                
                if error_type not in correlations:
                    correlations[error_type] = Counter()
                correlations[error_type][strategy] += 1
        
        # Convert Counter objects to simple dicts for the return type
        return {et: dict(counts) for et, counts in correlations.items()}


    def get_agent_performance_over_time(self, agent_type: AgentType, metric_name: str = "overall_score") -> List[Tuple[datetime, float]]:
        """
        Tracks a specific performance metric for an agent over time.
        """
        performance_series: List[Tuple[datetime, float]] = []
        for entry in self.history:
            if entry.agent_type == agent_type and entry.metrics:
                metric_value = getattr(entry.metrics, metric_name, None)
                if metric_value is not None:
                    performance_series.append((entry.timestamp, metric_value))
        return sorted(performance_series, key=lambda x: x[0])

    def plot_evolution_graph(self, output_path: str):
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Skipping graph generation.")
            return

        G = nx.DiGraph()
        for entry in self.history:
            node_id_str = str(entry.id)
            label = f"{entry.agent_type.value[:4]}\n{node_id_str[:4]}"
            G.add_node(node_id_str, label=label, agent_type=entry.agent_type.value, succeeded=entry.succeeded)

        for entry in self.history:
            if entry.parent_id:
                parent_node_id_str = str(entry.parent_id)
                child_node_id_str = str(entry.id)
                if G.has_node(parent_node_id_str) and G.has_node(child_node_id_str):
                    edge_color = "green" if entry.succeeded else "red"
                    G.add_edge(parent_node_id_str, child_node_id_str, color=edge_color)
        
        if not G.nodes():
            print("No data to plot for evolution graph.")
            plt.close('all') # Close any existing figures
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42) # Added seed for reproducibility
        
        node_colors_list = []
        for node_id_for_color in G.nodes():
            node_data = G.nodes[node_id_for_color]
            succeeded_val = node_data.get('succeeded')
            if succeeded_val is True: node_colors_list.append('lightgreen')
            elif succeeded_val is False: node_colors_list.append('salmon')
            else: node_colors_list.append('lightgrey')

        labels_dict = nx.get_node_attributes(G, 'label')
        edge_colors_list = [G[u][v]['color'] for u,v in G.edges()]

        nx.draw(G, pos, ax=ax, labels=labels_dict, with_labels=True, node_color=node_colors_list, 
                edge_color=edge_colors_list, node_size=2000, font_size=8, arrows=True)
        ax.set_title("Prompt Evolution Graph") # Use ax.set_title
        try:
            plt.savefig(output_path)
        finally:
            plt.close(fig) # Close specific figure

    def plot_effectiveness_heatmap(self, output_path: str):
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Skipping heatmap generation.")
            return

        effectiveness_data = self.get_modification_effectiveness()
        if not effectiveness_data:
            print("No data to plot for effectiveness heatmap.")
            plt.close('all')
            return

        # Prepare data for DataFrame
        strategies = list(effectiveness_data.keys())
        metrics_to_plot = ["success_rate", "total_applications", "successful_applications"]
        heatmap_values = []

        for strategy in strategies:
            row = [effectiveness_data[strategy].get(metric, 0) for metric in metrics_to_plot]
            heatmap_values.append(row)
        
        df = pd.DataFrame(heatmap_values, index=strategies, columns=metrics_to_plot)

        if df.empty:
            print("DataFrame for heatmap is empty. Skipping plot.")
            plt.close('all')
            return

        fig, ax = plt.subplots(figsize=(10, max(6, len(strategies) * 0.5)))
        seaborn.heatmap(df, ax=ax, annot=True, fmt=".2f", cmap="viridis")
        ax.set_title("Modification Strategy Effectiveness")
        ax.set_ylabel("Strategy")
        fig.tight_layout()
        try:
            plt.savefig(output_path)
        finally:
            plt.close(fig)

class TestEvolutionAnalytics(unittest.TestCase):
    def setUp(self):
        self.start_time = datetime.utcnow()
        self.entry1_id = uuid4()
        self.entry2_id = uuid4()
        self.entry3_id = uuid4()
        self.entry4_id = uuid4()
        self.entry5_id = uuid4()
        self.entry6_id = uuid4()
        self.entry7_id = uuid4()

        self.history_data: List[PromptHistoryEntry] = [
            PromptHistoryEntry(
                id=self.entry1_id, agent_type=AgentType.EXECUTOR, timestamp=self.start_time,
                generation_metadata={"operator_name": "add_try_except"},
                metrics=PromptEvaluationMetrics(overall_score=0.5, code_correctness=0.5),
                succeeded=False, parent_id=None
            ),
            PromptHistoryEntry(
                id=self.entry2_id, agent_type=AgentType.EXECUTOR, timestamp=self.start_time + timedelta(minutes=1),
                generation_metadata={"operator_name": "type_hint_fix"},
                metrics=PromptEvaluationMetrics(overall_score=0.7, code_correctness=0.7),
                succeeded=True, fixed_error_info={"original_error_type": ErrorType.TYPE_ERROR},
                parent_id=self.entry1_id 
            ),
            PromptHistoryEntry(
                id=self.entry3_id, agent_type=AgentType.PLANNER, timestamp=self.start_time + timedelta(minutes=2),
                generation_metadata={"operator_name": "clarify_steps"},
                metrics=PromptEvaluationMetrics(overall_score=0.6, logical_consistency=0.6),
                succeeded=False, parent_id=self.entry1_id
            ),
            PromptHistoryEntry(
                id=self.entry4_id, agent_type=AgentType.EXECUTOR, timestamp=self.start_time + timedelta(minutes=3),
                generation_metadata={"operator_name": "add_try_except"},
                metrics=PromptEvaluationMetrics(overall_score=0.8, code_correctness=0.8),
                succeeded=True, fixed_error_info={"original_error_type": ErrorType.SYNTAX_ERROR},
                parent_id=self.entry2_id 
            ),
            PromptHistoryEntry(
                id=self.entry5_id, agent_type=AgentType.PLANNER, timestamp=self.start_time + timedelta(minutes=4),
                generation_metadata={"operator_name": "add_examples"},
                metrics=PromptEvaluationMetrics(overall_score=0.75, logical_consistency=0.75),
                succeeded=True, fixed_error_info={"original_error_type": ErrorType.PLANNING_ERROR},
                parent_id=self.entry3_id 
            ),
            PromptHistoryEntry(
                id=self.entry6_id, agent_type=AgentType.EXECUTOR, timestamp=self.start_time + timedelta(minutes=5),
                generation_metadata={"operator_name": "type_hint_fix"},
                metrics=PromptEvaluationMetrics(overall_score=0.6, code_correctness=0.6),
                succeeded=False, fixed_error_info={"original_error_type": ErrorType.TYPE_ERROR},
                parent_id=self.entry4_id
            ),
            PromptHistoryEntry(
                id=self.entry7_id, agent_type=AgentType.EXECUTOR, timestamp=self.start_time + timedelta(minutes=6),
                generation_metadata={},
                metrics=PromptEvaluationMetrics(overall_score=0.9, code_correctness=0.9),
                succeeded=True, fixed_error_info={"original_error_type": "LOGIC_ERROR"},
                parent_id=self.entry6_id
            ),
        ]
        self.analytics = EvolutionAnalytics(self.history_data)

    def test_get_modification_effectiveness(self):
        effectiveness = self.analytics.get_modification_effectiveness()
        self.assertIn("add_try_except", effectiveness)
        self.assertEqual(effectiveness["add_try_except"]["total_applications"], 2)
        self.assertEqual(effectiveness["add_try_except"]["successful_applications"], 1)
        self.assertAlmostEqual(effectiveness["add_try_except"]["success_rate"], 0.5)
        self.assertIn("type_hint_fix", effectiveness)
        self.assertEqual(effectiveness["type_hint_fix"]["total_applications"], 2)
        self.assertEqual(effectiveness["type_hint_fix"]["successful_applications"], 1)
        self.assertAlmostEqual(effectiveness["type_hint_fix"]["success_rate"], 0.5)
        self.assertIn("clarify_steps", effectiveness)
        self.assertEqual(effectiveness["clarify_steps"]["total_applications"], 1)
        self.assertEqual(effectiveness["clarify_steps"]["successful_applications"], 0)
        self.assertAlmostEqual(effectiveness["clarify_steps"]["success_rate"], 0.0)
        self.assertIn("add_examples", effectiveness)
        self.assertEqual(effectiveness["add_examples"]["total_applications"], 1)
        self.assertEqual(effectiveness["add_examples"]["successful_applications"], 1)
        self.assertAlmostEqual(effectiveness["add_examples"]["success_rate"], 1.0)
        self.assertIn("unknown_operator", effectiveness)
        self.assertEqual(effectiveness["unknown_operator"]["total_applications"], 1)
        self.assertEqual(effectiveness["unknown_operator"]["successful_applications"], 1)
        self.assertAlmostEqual(effectiveness["unknown_operator"]["success_rate"], 1.0)

    def test_get_error_fix_correlation(self):
        correlations = self.analytics.get_error_fix_correlation()
        self.assertIn(ErrorType.TYPE_ERROR, correlations)
        self.assertEqual(correlations[ErrorType.TYPE_ERROR].get("type_hint_fix"), 1)
        self.assertIn(ErrorType.SYNTAX_ERROR, correlations)
        self.assertEqual(correlations[ErrorType.SYNTAX_ERROR].get("add_try_except"), 1)
        self.assertIn(ErrorType.PLANNING_ERROR, correlations)
        self.assertEqual(correlations[ErrorType.PLANNING_ERROR].get("add_examples"), 1)
        self.assertIn(ErrorType.LOGIC_ERROR, correlations)
        self.assertEqual(correlations[ErrorType.LOGIC_ERROR].get("unknown_operator"), 1)
        self.assertNotIn(ErrorType.AMBIGUOUS_ERROR, correlations)

    def test_get_agent_performance_over_time(self):
        executor_performance = self.analytics.get_agent_performance_over_time(AgentType.EXECUTOR, "overall_score")
        self.assertEqual(len(executor_performance), 5)
        self.assertEqual(executor_performance[0], (self.start_time, 0.5))
        self.assertEqual(executor_performance[1], (self.start_time + timedelta(minutes=1), 0.7))
        self.assertEqual(executor_performance[2], (self.start_time + timedelta(minutes=3), 0.8))
        self.assertEqual(executor_performance[3], (self.start_time + timedelta(minutes=5), 0.6))
        self.assertEqual(executor_performance[4], (self.start_time + timedelta(minutes=6), 0.9))
        planner_performance = self.analytics.get_agent_performance_over_time(AgentType.PLANNER, "logical_consistency")
        self.assertEqual(len(planner_performance), 2)
        self.assertEqual(planner_performance[0], (self.start_time + timedelta(minutes=2), 0.6))
        self.assertEqual(planner_performance[1], (self.start_time + timedelta(minutes=4), 0.75))
        executor_custom_perf = self.analytics.get_agent_performance_over_time(AgentType.EXECUTOR, "non_existent_metric")
        self.assertEqual(len(executor_custom_perf), 0)

    @unittest.skipUnless(PLOTTING_AVAILABLE, "Plotting libraries not installed")
    @mock.patch('tests.test_evolution_analytics.plt.savefig')
    @mock.patch('tests.test_evolution_analytics.nx.draw')
    def test_plot_evolution_graph(self, mock_nx_draw, mock_savefig):
        output_path = "test_evolution_graph.png"
        with mock.patch('tests.test_evolution_analytics.plt.subplots', return_value=(mock.MagicMock(), mock.MagicMock())) as mock_subplots: # Mock subplots
            self.analytics.plot_evolution_graph(output_path)
        
        mock_subplots.assert_called_once() # Check if subplots was called
        mock_nx_draw.assert_called_once()
        mock_savefig.assert_called_once_with(output_path)
        args, kwargs = mock_nx_draw.call_args
        graph_passed = args[0]
        self.assertIsInstance(graph_passed, nx.DiGraph)
        self.assertEqual(len(graph_passed.nodes()), len(self.history_data))
        expected_edges = sum(1 for entry in self.history_data if entry.parent_id is not None)
        self.assertEqual(len(graph_passed.edges()), expected_edges)

    @unittest.skipUnless(PLOTTING_AVAILABLE, "Plotting libraries not installed")
    @mock.patch('tests.test_evolution_analytics.plt.savefig')
    @mock.patch('tests.test_evolution_analytics.seaborn.heatmap')
    def test_plot_effectiveness_heatmap(self, mock_seaborn_heatmap, mock_savefig):
        output_path = "test_effectiveness_heatmap.png"
        # Mock fig and ax returned by plt.subplots
        mock_fig = mock.MagicMock()
        mock_ax = mock.MagicMock()
        with mock.patch('tests.test_evolution_analytics.plt.subplots', return_value=(mock_fig, mock_ax)) as mock_subplots:
            self.analytics.plot_effectiveness_heatmap(output_path)

        mock_subplots.assert_called_once() # Check if subplots was called
        mock_seaborn_heatmap.assert_called_once()
        mock_savefig.assert_called_once_with(output_path)
        args, kwargs = mock_seaborn_heatmap.call_args
        df_passed = args[0]
        self.assertIsInstance(df_passed, pd.DataFrame)
        effectiveness_data = self.analytics.get_modification_effectiveness()
        self.assertEqual(len(df_passed), len(effectiveness_data))
        self.assertEqual(list(df_passed.columns), ["success_rate", "total_applications", "successful_applications"])

if __name__ == '__main__':
    unittest.main() 