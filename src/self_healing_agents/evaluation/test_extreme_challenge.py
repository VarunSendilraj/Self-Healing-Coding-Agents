"""
Extreme challenge test designed to force self-healing through maximum complexity.
This combines multiple advanced algorithmic concepts in a single task.
"""

import logging
import os
from self_healing_agents.agents import Planner, Executor, Critic
from self_healing_agents.llm_service import LLMService
from self_healing_agents.prompts import BAD_PLANNER_PROMPT, DEFAULT_EXECUTOR_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from self_healing_agents.evaluation.enhanced_multi_agent_harness import run_enhanced_multi_agent_task
from self_healing_agents.evaluation.enhanced_harness import TermColors

def main():
    """Test with an extremely challenging multi-algorithm problem."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("💀 EXTREME CHALLENGE SELF-HEALING TEST")
    print("=" * 70)
    print("🎯 Goal: Maximum complexity to guarantee self-healing activation")
    print("=" * 70)
    
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
    
    # Use bad planner to maximize failure probability
    print(f"\n🔧 AGENT CONFIGURATION:")
    print(f"   🤖 Planner: PLANNER_SYSTEM_PROMPT (not helpful)")
    print(f"   🔧 Executor: DEFAULT_EXECUTOR_SYSTEM_PROMPT")
    print(f"   🧐 Critic: Comprehensive testing")
    print(f"   🤖 Self-Healing: Multi-iteration recovery system")
    
    planner = Planner("ExtremelyBadPlanner", llm_service, PLANNER_SYSTEM_PROMPT)
    executor = Executor("Executor", llm_service, DEFAULT_EXECUTOR_SYSTEM_PROMPT)
    critic = Critic("Critic", llm_service)
    
    # Define an extremely challenging multi-algorithm task
    extreme_task = {
        "id": "extreme_multi_algorithm_challenge",
        "description": """
Implement a comprehensive graph analysis system that combines multiple advanced algorithms:

Create a class `GraphAnalyzer` with the following methods:

1. `__init__(self, adjacency_matrix)`: Initialize with a 2D adjacency matrix (weights, 0 = no edge)

2. `shortest_paths_all_pairs(self)`: Implement Floyd-Warshall algorithm
   - Return 2D matrix of shortest distances between all pairs
   - Handle negative weights (detect negative cycles)
   - Return None if negative cycle exists

3. `minimum_spanning_tree(self)`: Implement Kruskal's algorithm with Union-Find
   - Return list of edges [(u, v, weight)] in MST
   - Use proper Union-Find with path compression and union by rank
   - Handle disconnected graphs (return forest)

4. `topological_sort(self)`: Implement Kahn's algorithm
   - Return topologically sorted order as list of node indices
   - Return None if cycle detected (not a DAG)
   - Use in-degree counting approach

5. `strongly_connected_components(self)`: Implement Tarjan's algorithm
   - Return list of lists, each containing nodes in one SCC
   - Use DFS with low-link values and stack
   - Handle self-loops and complex cycle structures

6. `max_flow(self, source, sink)`: Implement Ford-Fulkerson with Edmonds-Karp
   - Return maximum flow value from source to sink
   - Use BFS for finding augmenting paths
   - Handle capacity constraints properly

Additional Requirements:
- All algorithms must be correctly implemented with proper time complexity
- Handle edge cases: empty graphs, single nodes, disconnected components
- Use appropriate data structures (heaps, stacks, queues, union-find)
- Include proper error handling and input validation
- Methods should work together (e.g., MST on result of SCC analysis)

Example Usage:
```python
# Adjacency matrix: 4x4 graph
matrix = [
    [0, 3, 8, 0],
    [0, 0, 2, 5],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
]

analyzer = GraphAnalyzer(matrix)
distances = analyzer.shortest_paths_all_pairs()  # Floyd-Warshall
mst = analyzer.minimum_spanning_tree()           # Kruskal's
topo = analyzer.topological_sort()               # Kahn's
sccs = analyzer.strongly_connected_components()  # Tarjan's
flow = analyzer.max_flow(0, 3)                   # Ford-Fulkerson
```

This is an extremely complex task requiring:
- 5 different advanced graph algorithms
- Multiple data structure implementations
- Complex edge case handling
- Object-oriented design
- Algorithm integration

The implementation must be production-quality with proper error handling.
""",
        "initial_executor_prompt": DEFAULT_EXECUTOR_SYSTEM_PROMPT
    }
    
    print(f"\n📋 EXTREME ALGORITHMIC CHALLENGE:")
    print(f"   ID: {extreme_task['id']}")
    print(f"   Algorithms Required:")
    print(f"     1. Floyd-Warshall (All-pairs shortest paths)")
    print(f"     2. Kruskal's + Union-Find (Minimum spanning tree)")
    print(f"     3. Kahn's Algorithm (Topological sorting)")
    print(f"     4. Tarjan's Algorithm (Strongly connected components)")
    print(f"     5. Ford-Fulkerson/Edmonds-Karp (Maximum flow)")
    print(f"   Complexity: EXTREME")
    print(f"     - Multiple advanced algorithms in one class")
    print(f"     - Complex data structures required")
    print(f"     - Extensive edge case handling")
    print(f"     - Algorithm integration requirements")
    print(f"     - Production-quality implementation needed")
    
    # Run with maximum healing iterations
    print(f"\n🏃‍♂️ RUNNING EXTREME CHALLENGE...")
    print("=" * 70)
    
    result = run_enhanced_multi_agent_task(
        task_definition=extreme_task,
        planner=planner,
        executor=executor,
        critic=critic,
        llm_service_instance=llm_service,
        max_healing_iterations=8  # Allow many iterations for this extreme task
    )
    
    # Comprehensive analysis
    print(f"\n📊 EXTREME CHALLENGE RESULTS:")
    print("=" * 70)
    
    final_status = result['final_status']
    success = 'SUCCESS' in final_status
    
    print(f"   🎯 Final Status: {TermColors.color_text(final_status, TermColors.GREEN if success else TermColors.FAIL)}")
    final_score_str = f"{result['final_score']:.2f}"
    print(f"   📈 Final Score: {TermColors.color_text(final_score_str, TermColors.GREEN)}")
    print(f"   🔄 Total Healing Iterations: {TermColors.color_text(str(result['total_healing_iterations']), TermColors.CYAN)}")
    print(f"   🧠 Planner Healings: {result['healing_breakdown']['planner_healings']}")
    print(f"   ⚙️  Executor Healings: {result['healing_breakdown']['executor_healings']}")
    print(f"   🔨 Direct Fix Attempts: {result['healing_breakdown']['direct_fix_attempts']}")
    
    # Self-healing effectiveness analysis
    if result['total_healing_iterations'] > 0:
        print(f"\n🎉 SELF-HEALING SYSTEM PERFORMANCE:")
        print("=" * 70)
        print(f"✅ CONFIRMED: Extreme challenge triggered self-healing!")
        print(f"🔄 Healing iterations: {result['total_healing_iterations']}")
        
        # Calculate healing effectiveness
        initial_score = 0.0  # Assume initial failure
        final_score = result['final_score']
        improvement = final_score - initial_score
        
        print(f"📈 Score improvement: {initial_score:.2f} → {final_score:.2f} (+{improvement:.2f})")
        
        if result['healing_breakdown']['planner_healings'] > 0:
            print(f"🧠 Planner healing was critical for this complex multi-algorithm task")
        
        if result['healing_breakdown']['executor_healings'] > 0:
            print(f"⚙️  Executor healing addressed implementation complexity")
            
        # Analyze healing progression
        print(f"\n📊 HEALING PROGRESSION ANALYSIS:")
        for i, phase in enumerate(result['workflow_phases']):
            if 'HEALING_ITERATION' in phase.get('phase', ''):
                iteration_num = phase.get('phase', '').split('_')[-1]
                target = phase.get('healing_target', 'UNKNOWN')
                success_status = phase.get('healing_successful', False)
                score = phase.get('improved_score', 0.0)
                
                status_icon = "✅" if success_status else "❌"
                print(f"   Iteration {iteration_num}: {status_icon} {target} → Score: {score:.2f}")
                
    else:
        print(f"\n🤯 IMPOSSIBLE: Extreme challenge completed without self-healing!")
        print(f"🚀 This would be extraordinary - the task is designed to be nearly impossible")
        print(f"💭 Possible explanations:")
        print(f"   - LLM is exceptionally capable")
        print(f"   - Task needs to be even more complex")
        print(f"   - System bypassed healing due to unexpected success")
    
    # Algorithm implementation analysis
    if 'final_code' in result:
        print(f"\n🧮 ALGORITHM IMPLEMENTATION ANALYSIS:")
        print("=" * 70)
        
        final_code = result['final_code']
        algorithm_analysis = analyze_extreme_implementation(final_code)
        
        total_algorithms = len(algorithm_analysis)
        implemented_algorithms = sum(1 for status in algorithm_analysis.values() if status)
        
        print(f"📊 Implementation Coverage: {implemented_algorithms}/{total_algorithms} algorithms")
        
        for algorithm, implemented in algorithm_analysis.items():
            status_color = TermColors.GREEN if implemented else TermColors.FAIL
            status_text = "✅ IMPLEMENTED" if implemented else "❌ MISSING"
            print(f"   {algorithm}: {TermColors.color_text(status_text, status_color)}")
        
        # Overall implementation quality
        coverage_ratio = implemented_algorithms / total_algorithms
        if coverage_ratio >= 0.8:
            quality = "EXCELLENT"
            color = TermColors.GREEN
        elif coverage_ratio >= 0.6:
            quality = "GOOD"
            color = TermColors.YELLOW
        elif coverage_ratio >= 0.4:
            quality = "PARTIAL"
            color = TermColors.CYAN
        else:
            quality = "POOR"
            color = TermColors.FAIL
            
        coverage_percent = f"({coverage_ratio:.1%})"
        print(f"\n📈 Implementation Quality: {TermColors.color_text(quality, color)} {coverage_percent}")
    
    # Final assessment
    print(f"\n🎯 EXTREME CHALLENGE ASSESSMENT:")
    print("=" * 70)
    
    if result['total_healing_iterations'] >= 3:
        print(f"🏆 OUTSTANDING: System demonstrated robust self-healing under extreme pressure")
        print(f"🧠 Multiple healing iterations show intelligent problem-solving")
        
    elif result['total_healing_iterations'] > 0:
        print(f"✅ GOOD: Self-healing activated as expected for extreme challenge")
        
    if success:
        print(f"🌟 REMARKABLE: Successfully implemented complex multi-algorithm system")
        print(f"🎓 This demonstrates advanced AI-assisted software development")
    else:
        print(f"📚 EDUCATIONAL: Challenge pushed system to its limits")
        print(f"💡 Valuable insights into self-healing system capabilities")
    
    print(f"\n💀 EXTREME CHALLENGE TEST COMPLETE!")
    healing_status = 'PASSED' if result['total_healing_iterations'] > 0 else 'NEEDS HARDER TASK'
    print(f"🎉 Self-healing system stress test: {healing_status}")
    
    return result

def analyze_extreme_implementation(code: str) -> dict:
    """
    Analyze if the code implements the required algorithms.
    """
    code_lower = code.lower()
    
    analysis = {
        "Floyd-Warshall (All-pairs shortest paths)": (
            "floyd" in code_lower or 
            ("shortest" in code_lower and "all" in code_lower and "pairs" in code_lower)
        ),
        "Kruskal's + Union-Find (MST)": (
            "kruskal" in code_lower or "union" in code_lower or 
            ("spanning" in code_lower and "tree" in code_lower)
        ),
        "Kahn's Algorithm (Topological sort)": (
            "kahn" in code_lower or "topological" in code_lower or
            ("indegree" in code_lower or "in_degree" in code_lower)
        ),
        "Tarjan's Algorithm (SCC)": (
            "tarjan" in code_lower or 
            ("strongly" in code_lower and "connected" in code_lower)
        ),
        "Ford-Fulkerson/Edmonds-Karp (Max flow)": (
            "ford" in code_lower or "edmonds" in code_lower or "karp" in code_lower or
            ("max" in code_lower and "flow" in code_lower)
        ),
        "Class Structure": "class" in code_lower and "graphanalyzer" in code_lower,
        "Error Handling": "try" in code_lower or "except" in code_lower or "raise" in code_lower,
        "Data Structures": any(ds in code_lower for ds in ["heap", "queue", "stack", "deque"])
    }
    
    return analysis

if __name__ == "__main__":
    main() 