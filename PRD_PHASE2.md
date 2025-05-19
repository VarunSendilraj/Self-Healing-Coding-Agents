# Product Requirements Document (PRD): 
## Self-Healing Agentic AI - Phase 2: Targeted Self-Healing with Error-Type Awareness

---

## 1. Introduction

### 1.1 Overview
This document outlines Phase 2 of the Self-Healing Agentic AI system, introducing targeted self-healing based on error type classification. Building upon the MVP's foundation, this phase enhances the system's ability to distinguish between planning and execution failures, routing them to the appropriate agent for correction.

### 1.2 Connection to Phase 1
Phase 1 established the core self-healing loop with a unified prompt modification approach. Phase 2 extends this by:
- Adding fine-grained error classification
- Implementing targeted prompt evolution
- Enhancing the Critic's analytical capabilities

### 1.3 Business/Technical Justification
- **Improved Efficiency**: Reduces unnecessary prompt modifications by targeting the root cause
- **Better Outcomes**: Addresses logical flaws at the planning level rather than patching execution
- **Reduced LLM Costs**: Fewer iterations needed by addressing the right problem area
- **More Maintainable**: Clearer separation of concerns between planning and execution
- **Higher Quality**: Test-driven development approach ensures robustness and correctness

---

## 2. Error Type Classification

### 2.1 Error Taxonomy

#### Planning Failures
- **Definition**: Flaws in the overall approach or strategy
- **Examples**:
  - Inconsistent or contradictory steps
  - Missing critical algorithmic components
  - Incorrect problem decomposition
  - Flawed logic flow
  - Inefficient solution approaches

#### Execution Failures
- **Definition**: Issues in code implementation
- **Examples**:
  - Syntax errors
  - Runtime exceptions
  - Off-by-one errors
  - Boundary condition issues
  - Incorrect library usage

### 2.2 Detection Mechanisms

#### Planning Failure Indicators
- Logical inconsistencies in generated code
- Missing edge case handling
- Inefficient algorithms
- Contradictions with requirements

#### Execution Failure Indicators
- Syntax errors
- Runtime exceptions
- Test case failures
- Performance issues

---

## 3. System Architecture

### 3.1 Enhanced Components

#### Critic Agent
- **Enhanced Error Analysis**: Classifies errors into planning/execution categories
- **Confidence Scoring**: Rates certainty of classification
- **Feedback Enrichment**: Provides targeted feedback based on error type

#### Prompt Modification System
- **Dual-Path Evolution**:
  - Planner-focused prompt evolution
  - Executor-focused prompt evolution
- **Context-Aware Updates**: Modifies prompts based on error context

### 3.2 Workflow Modifications

1. **Initial Execution**
   - Planner → Executor → Critic flow as before

2. **Failure Analysis**
   - Critic classifies error type
   - Determines confidence level

3. **Targeted Self-Healing**
   - High-confidence planning failures → Update Planner prompt
   - Execution failures → Update Executor prompt
   - Ambiguous cases → Conservative approach (default to Executor)

4. **Iterative Refinement**
   - Track which prompt modifications resolve which issues
   - Update classification heuristics based on outcomes

---

## 4. Technical Specifications

### 4.1 New Components

#### Error Classifier
```python
class ErrorClassifier:
    def classify_error(self, error_details: Dict) -> Tuple[ErrorType, float]:
        """Classifies error and returns type with confidence score"""
        pass
```

#### Enhanced Critic Report
```python
class EnhancedCriticReport(CriticReport):
    error_type: ErrorType
    confidence: float
    suggested_agent: AgentType  # PLANNER or EXECUTOR
    reasoning: str
```

### 4.2 Data Flow

1. **Error Detection**
   - Critic captures error details
   - Passes to Error Classifier

2. **Classification**
   - Error Classifier analyzes patterns
   - Returns classification with confidence

3. **Prompt Routing**
   - High confidence → Route to appropriate agent
   - Low confidence → Default to Executor

4. **Prompt Evolution**
   - Agent-specific prompt templates
   - Targeted modifications based on error type

### 4.3 Error Categorization Logic

#### Planning Failure Indicators
- Multiple logical contradictions
- Missing core algorithm steps
- Inefficient time/space complexity
- Incorrect problem decomposition

#### Execution Failure Indicators
- Syntax errors
- Runtime exceptions
- Type mismatches
- Library-specific errors

---

## 5. Implementation Plan

### 5.0 Development Methodology

This implementation will strictly follow a **Test-Driven Development (TDD)** approach. For each component:

1. **Write Tests First**: Before implementing any functionality, comprehensive tests will be written that define the expected behavior.
2. **Run Tests (They Should Fail)**: Verify that tests properly fail before implementation.
3. **Implement Minimal Code**: Write just enough code to make tests pass.
4. **Run Tests Again**: Ensure all tests now pass.
5. **Refactor**: Clean up the implementation while maintaining test coverage.

This approach ensures that:
- All code has corresponding tests from the beginning
- Implementation meets specified requirements
- Regressions are caught immediately
- Code maintains high quality and modularity

All new code must have ≥90% test coverage, with critical components targeting 100% coverage.

### 5.1 Phase 1: Core Classification (Weeks 1-2)
- Implement Error Classifier
- Enhance Critic with classification
- Basic prompt routing

### 5.2 Phase 2: Enhanced Prompt Evolution (Weeks 3-4)
- Agent-specific prompt templates
- Targeted modification strategies
- Confidence-based routing

### 5.3 Phase 3: Testing & Refinement (Weeks 5-6)
- Expansion of test-driven development approach
- Comprehensive unit and integration tests
- Performance benchmarking
- Test coverage analysis

### 5.4 Dependencies
- Phase 1 MVP completion
- Enhanced test suite
- Monitoring infrastructure

---

## 6. Success Metrics

### 6.1 Primary KPIs
- **Classification Accuracy**: >90% correct error type identification
- **Self-Healing Success Rate**: Improvement over Phase 1
- **Iterations to Resolution**: Reduction in average attempts

### 6.2 Secondary Metrics
- Confidence score distribution
- Error type distribution
- Prompt modification effectiveness

### 6.3 Evaluation Methodology
- A/B testing against Phase 1
- Controlled benchmark tasks
- Real-world scenario testing

---

## 7. Future Work

### 7.1 Short-term
- Fine-grained error subcategories
- Dynamic confidence thresholds
- Hybrid prompt modifications

### 7.2 Long-term
- Automated test case generation
- Multi-agent collaboration
- Online learning from corrections

### 7.3 Research Directions
- Few-shot learning for classification
- Transfer learning between tasks
- Meta-learning for prompt evolution

---

## 8. Risks & Mitigations

### 8.1 Classification Errors
- **Risk**: Incorrect error routing
- **Mitigation**: Confidence thresholds, fallback mechanisms

### 8.2 Prompt Drift
- **Risk**: Over-optimization for specific errors
- **Mitigation**: Diversity preservation, periodic resets

### 8.3 Performance Impact
- **Risk**: Increased latency from analysis
- **Mitigation**: Caching, optimizations

---

## 9. Conclusion

This Phase 2 enhancement will significantly improve the system's ability to self-heal by targeting the root cause of failures. By distinguishing between planning and execution errors, we can apply more effective corrections and build a more robust, maintainable system.

---

## 10. Implementation Task List

This section provides a detailed breakdown of tasks required to implement the Phase 2 functionality based on the current codebase.

### 10.1 Core Schema Updates (Week 1)

* [ ] Task 1.1: **Define Error Type Taxonomy**
    * Write tests for `error_types.py` that validate enum behavior and expected constants
    * Create `error_types.py` module with enums and constants
    * Define `ErrorType` enum (`PLANNING_ERROR`, `EXECUTION_ERROR`, `AMBIGUOUS_ERROR`)
    * Define `AgentType` enum (`PLANNER`, `EXECUTOR`)
    * Define confidence thresholds for classification
    * Verify all tests pass

* [ ] Task 1.2: **Enhance Critic Report Schema**
    * Write tests for enhanced `CriticReport` schema in `tests/test_schemas.py`
    * Update `src/self_healing_agents/schemas.py` to extend `CriticReport`
    * Add fields for `error_type`, `confidence`, `suggested_agent`, and `reasoning`
    * Implement backward compatibility for existing code
    * Update JSON serialization/deserialization methods
    * Ensure tests validate all new fields and backwards compatibility

* [ ] Task 1.3: **Create Error Classification Interfaces**
    * Write tests for `ErrorClassifier` interface in `tests/test_classifiers.py`
    * Define `ErrorClassifier` interface in `src/self_healing_agents/classifiers.py`
    * Define methods for error classification with confidence scoring
    * Create base implementation with default classification logic
    * Verify interface implementation with test cases

### 10.2 Critic Agent Enhancement (Week 1-2)

* [ ] Task 2.1: **Update Critic Agent Core Logic**
    * Write tests for enhanced Critic functionality in `tests/test_critic.py`
    * Modify `src/self_healing_agents/agents.py` Critic class
    * Enhance error analysis capabilities in `_execute_sandboxed_code`
    * Add new method `classify_error_type` for determining error source
    * Update `run` method to include error classification in output
    * Verify all tests pass with the enhanced functionality

* [ ] Task 2.2: **Implement Rule-Based Error Classifier**
    * Write comprehensive tests for rule-based classification with various error scenarios
    * Create `src/self_healing_agents/classifiers/rule_based.py`
    * Implement pattern matching for common syntax/runtime errors
    * Develop heuristics for logical inconsistencies
    * Add confidence scoring based on rule strength
    * Validate with test suite including edge cases

* [ ] Task 2.3: **LLM-Based Error Classifier**
    * Write tests for LLM-based classifier with mock LLMService responses
    * Create `src/self_healing_agents/classifiers/llm_based.py`
    * Develop specialized prompts for error analysis
    * Implement LLM-based classification using existing LLMService
    * Design feedback format for classification reasoning
    * Test with representative error cases

* [ ] Task 2.4: **Hybrid Classification System**
    * Write tests for hybrid classification system covering various scenarios
    * Create orchestration layer for combining rule-based and LLM classifiers
    * Implement weighted voting or hierarchical fallback
    * Add confidence calculation for final classification
    * Develop logging and explanation generation
    * Ensure high test coverage for decision logic

### 10.3 Prompt Modification System Enhancements (Week 2-3)

* [ ] Task 3.1: **Extend PromptModifier for Multi-Agent Support**
    * Update `src/self_healing_agents/prompt_modifier.py`
    * Add support for tracking both Planner and Executor prompts
    * Implement separate evolution paths for each agent type
    * Modify population management to handle dual-path evolution

* [ ] Task 3.2: **Agent-Specific Evolution Operators**
    * Create specialized evolution operators for Planner prompts
    * Create specialized evolution operators for Executor prompts
    * Implement context-aware operator selection
    * Add methods for targeted prompt generation

* [ ] Task 3.3: **Prompt History and Tracking**
    * Implement enhanced prompt history tracking
    * Add correlation between error types and successful fixes
    * Develop metrics for evolution effectiveness by agent type
    * Create visualization tools for prompt evolution paths

* [ ] Task 3.4: **Confidence-Based Routing Logic**
    * Implement logic for determining which agent to modify
    * Add fallback mechanisms for low-confidence cases
    * Create hybrid modification strategies for ambiguous errors
    * Develop adaptive confidence thresholds

### 10.4 Orchestration Updates (Week 3-4)

* [ ] Task 4.1: **Update Orchestrator for Targeted Self-Healing**
    * Modify `src/self_healing_agents/orchestration.py` and `orchestrator.py`
    * Enhance `should_trigger_self_healing` to handle error types
    * Add support for targeted agent selection
    * Implement differentiated healing loops

* [ ] Task 4.2: **Create Enhanced Task Factory Methods**
    * Update `src/self_healing_agents/tasks.py`
    * Add new task types for targeted self-healing
    * Implement agent-specific task parameters
    * Create specialized critique tasks with error type awareness

* [ ] Task 4.3: **Workflow Management Enhancements**
    * Implement workflow branching based on error classification
    * Add support for parallel prompt evolution
    * Create checkpointing for multi-path exploration
    * Develop iteration limits per agent type

* [ ] Task 4.4: **API and Interface Updates**
    * Enhance public interfaces to expose targeted healing
    * Update method signatures to support new functionality
    * Ensure backward compatibility
    * Add new convenience methods for common operations

### 10.5 Testing and Evaluation Infrastructure (Week 4-5)

> Note: While testing tasks are grouped here, remember that per our TDD approach, tests for each component should be written *before* the implementation of that component in earlier phases.

* [ ] Task 5.1: **Enhanced Test Coverage**
    * Ensure 90%+ test coverage for all new components
    * Create comprehensive `tests/test_error_classifier.py` with all edge cases
    * Update `tests/test_critic.py` for new functionality
    * Add test cases for targeted healing
    * Develop benchmark problems with known error types
    * Implement coverage reporting in CI pipeline

* [ ] Task 5.2: **Evaluation Framework Updates**
    * Enhance `src/self_healing_agents/evaluation/harness.py`
    * Add metrics for classification accuracy
    * Implement comparative assessment (Phase 1 vs. Phase 2)
    * Create visualizations for performance analysis

* [ ] Task 5.3: **End-to-End System Tests**
    * Develop integration tests for full self-healing loop
    * Create simulation framework for error injection
    * Implement A/B testing capability
    * Add stress testing for edge cases

* [ ] Task 5.4: **Baseline Establishment**
    * Run Phase 1 system on benchmark problems
    * Document performance metrics
    * Establish comparison methodology
    * Create baseline for improvement measurement

### 10.6 Documentation and Deployment (Week 5-6)

* [ ] Task 6.1: **Code Documentation**
    * Update docstrings for all new and modified classes
    * Create architectural documentation
    * Update module-level documentation
    * Add examples and usage guides

* [ ] Task 6.2: **User Documentation**
    * Create user guide for targeted self-healing
    * Document configuration options
    * Add troubleshooting section
    * Provide examples of common usage patterns

* [ ] Task 6.3: **System Integration**
    * Ensure compatibility with existing workflows
    * Implement feature flags for gradual rollout
    * Add telemetry for performance monitoring
    * Create migration guide from Phase 1

* [ ] Task 6.4: **Final Testing and Deployment**
    * Conduct final QA testing
    * Address any outstanding issues
    * Prepare release notes
    * Deploy to production environment

### 10.7 Monitoring and Optimization (Ongoing)

* [ ] Task 7.1: **Performance Monitoring**
    * Implement metrics collection
    * Create dashboards for key indicators
    * Set up alerts for degradation
    * Establish performance baselines

* [ ] Task 7.2: **Continuous Improvement**
    * Analyze usage patterns
    * Identify common failure modes
    * Refine classification heuristics
    * Optimize resource usage

* [ ] Task 7.3: **Feedback Collection**
    * Create mechanism for capturing feedback
    * Implement automated analysis of results
    * Develop iterative improvement process
    * Document lessons learned

---
