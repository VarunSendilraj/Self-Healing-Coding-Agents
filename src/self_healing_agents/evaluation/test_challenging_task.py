"""
Test script with a challenging algorithmic problem designed to trigger self-healing.
This uses a complex graph algorithm that requires sophisticated planning and implementation.
"""

import logging
import os
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import BAD_PLANNER_PROMPT, DEFAULT_EXECUTOR_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from self_healing_agents.evaluation.enhanced_multi_agent_harness import run_enhanced_multi_agent_task
from self_healing_agents.evaluation.enhanced_harness import TermColors

def main():
    """Test with a very challenging algorithmic problem that should trigger self-healing."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🔥 CHALLENGING SELF-HEALING TEST")
    print("=" * 60)
    print("🎯 Goal: Force the system to use self-healing with a complex algorithm")
    print("=" * 60)
    
    # Configure environment
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model_name = os.environ.get("LLM_MODEL", "deepseek-coder")
    
    # Initialize LLM service
    try:
        llm_service = LLMService(provider=provider, model_name=model_name)
        print(f"✅ LLM Service: {provider}/{model_name}")
    except Exception as e:
        print(f"❌ LLM Service Error: {e}")
        return
    
    # Use a deliberately vague planner to increase failure likelihood
    print(f"\n🔧 AGENT CONFIGURATION:")
    print(f"   🤖 Planner: BAD_PLANNER_PROMPT (intentionally vague)")
    print(f"   🔧 Executor: DEFAULT_EXECUTOR_SYSTEM_PROMPT")
    print(f"   🧐 Critic: Standard evaluation with comprehensive tests")
    print(f"   🤖 Self-Healing: LLM-based intelligent recovery")
    
    # Use the bad planner to increase chances of needing self-healing
    planner = Planner("BadPlanner", llm_service, BAD_PLANNER_PROMPT)
    executor = Executor("Executor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)
    critic = Critic("Critic", llm_service)
    
    # Define a very challenging algorithmic task
    challenging_task = {
        "id": "challenging_graph_algorithm",
        "description": """
Implement Dijkstra's shortest path algorithm with the following specific requirements:

You are given a weighted directed graph represented as an adjacency list where each node maps to a list of (neighbor, weight) tuples. Implement a function `dijkstra_shortest_path(graph, start, end)` that:

1. Finds the shortest path from start node to end node using Dijkstra's algorithm
2. Returns a tuple: (shortest_distance, path_as_list_of_nodes)
3. If no path exists, return (float('inf'), [])
4. Handle edge cases: invalid start/end nodes, empty graph, self-loops
5. Use a proper priority queue (heapq) for efficiency
6. Track the actual path, not just distances

Additional Requirements:
- The graph input format: {"A": [("B", 5), ("C", 3)], "B": [("D", 2)], ...}
- Start and end are string node identifiers
- Weights are positive integers or floats
- Must handle disconnected components
- Must return the exact path taken, not just the distance

Example:
graph = {
    "A": [("B", 4), ("C", 2)],
    "B": [("C", 1), ("D", 5)], 
    "C": [("D", 8), ("E", 10)],
    "D": [("E", 2)],
    "E": []
}

dijkstra_shortest_path(graph, "A", "E") should return (8, ["A", "C", "D", "E"])
dijkstra_shortest_path(graph, "A", "F") should return (float('inf'), [])

The algorithm must be correct, efficient, and handle all edge cases properly.
""",
        "initial_executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT
    }
    
    print(f"\n📋 CHALLENGING ALGORITHMIC TASK:")
    print(f"   ID: {challenging_task['id']}")
    print(f"   Algorithm: Dijkstra's Shortest Path with Path Reconstruction")
    print(f"   Complexity: High - requires proper data structures, edge case handling")
    print(f"   Expected Challenges:")
    print(f"     - Complex algorithm implementation")
    print(f"     - Priority queue usage")
    print(f"     - Path reconstruction logic")
    print(f"     - Multiple edge cases")
    print(f"     - Specific input/output format requirements")
    
    # Run the enhanced harness
    print(f"\n🏃‍♂️ RUNNING ENHANCED MULTI-AGENT HARNESS...")
    print("=" * 60)
    
    result = run_enhanced_multi_agent_task(
        task_definition=challenging_task,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=5  # Allow more iterations for this complex task
    )
    
    # Comprehensive results analysis
    print(f"\n📊 COMPREHENSIVE RESULTS ANALYSIS:")
    print("=" * 60)
    
    final_status = result['final_status']
    success = 'SUCCESS' in final_status
    
    print(f"   🎯 Final Status: {TermColors.color_text(final_status, TermColors.GREEN if success else TermColors.FAIL)}")
    final_score_str = f"{result['final_score']:.2f}"
    print(f"   📈 Final Score: {TermColors.color_text(final_score_str, TermColors.GREEN)}")
    print(f"   🔄 Total Healing Iterations: {result['total_healing_iterations']}")
    print(f"   🧠 Planner Healings: {result['healing_breakdown']['planner_healings']}")
    print(f"   ⚙️  Executor Healings: {result['healing_breakdown']['executor_healings']}")
    print(f"   🔨 Direct Fix Attempts: {result['healing_breakdown']['direct_fix_attempts']}")
    
    # Detailed Self-Healing Analysis
    if result['total_healing_iterations'] > 0:
        print(f"\n🔧 SELF-HEALING SYSTEM ANALYSIS:")
        print("=" * 60)
        print(f"✅ SUCCESS: Self-healing system was activated!")
        print(f"🧠 The system made {result['total_healing_iterations']} healing attempt(s)")
        
        # Analyze what types of healing occurred
        if result['healing_breakdown']['planner_healings'] > 0:
            print(f"📋 Planner was healed {result['healing_breakdown']['planner_healings']} time(s)")
            print(f"   → This indicates the original plan was insufficient for the complex algorithm")
        
        if result['healing_breakdown']['executor_healings'] > 0:
            print(f"⚙️  Executor was healed {result['healing_breakdown']['executor_healings']} time(s)")
            print(f"   → This indicates implementation issues with the complex algorithm")
            
        if result['healing_breakdown']['direct_fix_attempts'] > 0:
            print(f"🔨 Direct fixes attempted {result['healing_breakdown']['direct_fix_attempts']} time(s)")
            print(f"   → This shows the system tried quick fixes before full healing")
    else:
        print(f"\n⚠️  UNEXPECTED: Self-healing was not triggered!")
        print(f"💡 This could mean:")
        print(f"   - The task succeeded on first try (unlikely with bad planner)")
        print(f"   - The system needs an even more challenging task")
        print(f"   - The bad planner prompt needs to be more problematic")
    
    # LLM Classification Analysis
    if result.get('classification_history'):
        print(f"\n🤖 LLM FAILURE CLASSIFICATION ANALYSIS:")
        print("=" * 60)
        
        for i, classification in enumerate(result['classification_history'], 1):
            failure_type = classification['primary_failure_type']
            confidence = classification['confidence']
            target = classification['recommended_healing_target']
            
            print(f"\n📋 Classification {i}:")
            print(f"   🔸 Failure Type: {TermColors.color_text(failure_type, TermColors.CYAN)}")
            print(f"   🔸 Confidence: {TermColors.color_text(f'{confidence:.2f}', TermColors.GREEN)}")
            print(f"   🔸 Target: {TermColors.color_text(target, TermColors.YELLOW)}")
            
            if classification.get("reasoning"):
                print(f"   🧠 LLM Reasoning:")
                for j, reason in enumerate(classification["reasoning"], 1):
                    print(f"      {j}. {reason}")
    
    # Algorithm-Specific Analysis
    print(f"\n🧮 ALGORITHM-SPECIFIC ANALYSIS:")
    print("=" * 60)
    
    # Check if the final solution properly implements Dijkstra's
    final_phase = result['workflow_phases'][-1] if result['workflow_phases'] else {}
    
    if 'final_code' in result:
        final_code = result['final_code']
        algorithm_analysis = analyze_dijkstra_implementation(final_code)
        
        print(f"📊 Dijkstra's Implementation Analysis:")
        for aspect, status in algorithm_analysis.items():
            status_color = TermColors.GREEN if status else TermColors.FAIL
            status_text = "✅ PRESENT" if status else "❌ MISSING"
            print(f"   {aspect}: {TermColors.color_text(status_text, status_color)}")
    
    # Final Assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    print("=" * 60)
    
    if result['total_healing_iterations'] > 0:
        print(f"🎉 EXCELLENT: The challenging task successfully triggered self-healing!")
        print(f"🧠 System demonstrated intelligent failure recovery")
        print(f"📈 Final outcome: {final_status} with score {result['final_score']:.2f}")
        
        if success:
            print(f"✨ OUTSTANDING: Complex algorithm was successfully implemented through self-healing")
        else:
            print(f"📚 LEARNING: Even with self-healing, the task remained challenging")
            print(f"💡 This demonstrates the system's limits and areas for improvement")
    else:
        print(f"🤔 INTERESTING: Task completed without self-healing")
        if success:
            print(f"🚀 IMPRESSIVE: Complex algorithm solved on first attempt!")
        else:
            print(f"💭 CONSIDER: Task may need to be even more challenging")
    
    print(f"\n🎉 CHALLENGING SELF-HEALING TEST COMPLETE!")
    return result

def analyze_dijkstra_implementation(code: str) -> dict:
    """
    Analyze if the code properly implements Dijkstra's algorithm.
    Returns a dict of implementation aspects and whether they're present.
    """
    code_lower = code.lower()
    
    analysis = {
        "Priority Queue (heapq)": "heapq" in code_lower or "priorityqueue" in code_lower,
        "Distance Tracking": "distance" in code_lower or "dist" in code_lower,
        "Path Reconstruction": "path" in code_lower and ("parent" in code_lower or "previous" in code_lower),
        "Visited Set": "visited" in code_lower or "seen" in code_lower,
        "Graph Traversal": "neighbor" in code_lower or "edge" in code_lower,
        "Infinity Handling": "inf" in code_lower or "float('inf')" in code,
        "Return Format": "return" in code_lower and ("tuple" in code_lower or "," in code),
        "Edge Case Handling": ("if" in code_lower and "not" in code_lower) or "empty" in code_lower
    }
    
    return analysis

if __name__ == "__main__":
    main() 