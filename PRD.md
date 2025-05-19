# Product Requirements Document (PRD): Self-Healing Agentic AI MVP

## 1. Introduction

---

## File Creation and Project Structure Guidelines

To ensure maintainability and clarity, all new files and modules should follow the structured layout below:

### 1. Source Code Organization
- **All core source files must be placed under:**
  - `src/self_healing_agents/`
- **Module Naming:**
  - Each Python module (file) should have a clear, descriptive name (e.g., `agents.py`, `prompt_modifier.py`).
  - Use lowercase with underscores for filenames.
- **Class and Function Naming:**
  - Use `CamelCase` for classes and `snake_case` for functions, following PEP 8.

### 2. Import Statements
- **All local imports must be absolute and prefixed with** `self_healing_agents.` (e.g., `from self_healing_agents.tools import TestRunnerTool`).
- **Do not use relative imports** (e.g., `from .tools import ...`).
- **External dependencies** should be imported as usual (e.g., `import torch`).

### 3. Test Files
- **All tests must be placed under:**
  - `tests/`
- **Test Naming:**
  - Test files should start with `test_` and mirror the module they are testing (e.g., `test_prompt_modifier.py`).
  - Each test file should import from `self_healing_agents.` using the absolute import style.
- **Test Driven Development:**
  - All new code should be accompanied by tests.
  - Tests should be written before the code they are testing.
  - Tests should be written in a way that is easy to understand and easy to write.
  - Tests should be written in a way that is easy to run and easy to debug.
  - Tests should be written in a way that is easy to maintain.
  - Tests should be written in a way that is easy to scale.

### 4. Configuration and Utility Files
- **Configuration files** (e.g., `.env`, `config.py`) should be placed at the project root or in `src/self_healing_agents/` if tightly coupled to the codebase.
- **Utility scripts** for data, logging, or setup should reside in `src/self_healing_agents/` or a subdirectory such as `src/self_healing_agents/utils/`.

### 5. Adding New Files
- **New agent, tool, or module:** Create a new `.py` file in `src/self_healing_agents/` and ensure all imports use the `self_healing_agents.` prefix.
- **New test:** Create a corresponding `test_*.py` file in `tests/` and import the target module using the absolute import style.
- **Submodules:** For logical grouping, create a subdirectory under `src/self_healing_agents/` with an `__init__.py`.

### 6. Example
```
SelfHealingAgents/
├── src/
│   └── self_healing_agents/
│       ├── ...
├── tests/
│   ├── test_...
│   └── ...
├── .env
├── PRD.md
└── ...
```

---

### 1.1 Problem Statement

Current multi-agent Large Language Model (LLM) systems, while promising for complex tasks, lack robustness in the face of failure. When an agent provides incorrect output, misinterprets instructions, or encounters unexpected conditions, the system typically halts or requires external human intervention. This fragility prevents the deployment of such systems in dynamic or long-horizon tasks where autonomous recovery is critical.

### 1.2 Goal

The goal of this MVP is to build and demonstrate a minimal viable version of a self-healing agentic AI system. This MVP will focus on the domain of Python code generation and debugging, showcasing the core loop where agents detect failure and autonomously modify their instructions (prompts) using a constrained evolutionary optimization approach to attempt recovery and achieve a better outcome.

### 1.3 MVP Scope

This MVP is specifically scoped to address:

* **Programming Language:** Python
* **Task Complexity:** Generating and debugging functions or scripts.
* **Error Triggering:** The system should be capable of detecting and being *triggered* by common Python errors within the MVP task scope, including syntax errors, basic runtime errors (e.g., division by zero, index errors), and failures against simple logical tests.
* **Self-Healing Mechanism:** Implementation of a *discrete* (text-based) prompt modification strategy based on evolutionary optimization algorithm (EvoPrompt).
* **Evaluation:** The Critic agent will perform implicit judgment augmented by generating and running simple test cases to provide quantifiable feedback used by the self-healing mechanism.

## 2. Goals

The specific, measurable goals for the Self-Healing Agentic AI MVP are:

* Successfully demonstrate the execution of the full agentic self-healing loop (Planner -> Executor -> Critic -> Failure Detection -> Prompt Modification via Constrained EvoPrompt -> Loop back to Executor) on a predefined set of small Python coding tasks.
* Implement the Critic agent's capability to run Python code, capture errors (syntax, runtime), and generate/run simple test cases against the code, providing structured feedback and a quantifiable score.
* Implement the EvoPrompt algorithm within the Prompt Modifier module to generate a revised discrete prompt for the Executor based on the Critic's failure feedback and evaluation score.
* Show quantifiable improvement in the generated code's correctness (measured by the Critic's generated test pass rate score) after one or more self-healing iterations in a majority of test cases where the initial attempt failed.
* Successfully integrate the chosen agent orchestration framework (Crew AI or alternative) and selected LLM provider (Groq/Hugging Face API) to support the dynamic prompt modification and LLM-based "Evolution" step required by the EvoPrompt process.

## 3. User Stories (Implicit System Behaviors)

* As the system, when given a request to write a small Python function, I should generate the initial code.
* As the system, when my initial code fails during execution or against a test, I should detect the failure and receive structured feedback from my evaluation component.
* As the system, upon receiving failure feedback, I should use this information within a prompt optimization process to generate a better hollistic prompt for my code-writing component (executor agent).
* As the system, I should attempt to fix the code based on the newly optimized prompt.
* As the system, I should track the results and prompt modifications across attempts for a given task instance.

## 4. System Architecture

The MVP system will consist of three primary agents and a distinct Prompt Modification module implementing a constrained EvoPrompt process, orchestrated to execute a task and attempt self-correction upon failure.

```mermaid
graph TD
    subgraph MVP Self-Healing Code Debugging Flow
    A[Start Task/Goal] --> B(Planner Agent: Understand Goal<br/>Outline Code Structure);
    B --> C(Executor Agent: Generate Python Code);
    C --> D(Critic Evaluation);
    D --> E{Failure Detected?};
    E -->|No - Code Passes Tests| F[Critic: Final Review/Success];
    F --> G[Task Success];
    E -->|Yes - Code Fails Tests/Errors| H(Prompt Self-Healing);
    H --> C; % Loop back to Executor with revised prompt
    end

    subgraph Critic Evaluation
    D --> D1(Receive Code from Executor);
    D1 --> D2(Attempt to Run Code<br/>& Capture Errors<br/>(Syntax/Runtime));
    D2 --> D3(Generate Simple Test Case<br/>Based on Task/Code Description);
    D3 --> D4(Run Code against Generated Test<br/>& Capture Results);
    D4 --> D5(Analyze Errors & Test Results<br/>Formulate Feedback & Score);
    D5 --> E; % Connects back to Failure Detected?
    end

    subgraph Prompt Self-Healing (Constrained EvoPrompt MVP)
    H --> H1(Receive Failure Report<br/>& Score from Critic);
    H1 --> H2(Initialize/Update Small Runtime Population<br/>(e.g., current prompt + history));
    H2 --> H3a(Select Parent Prompts<br/>from Population);
    H3a --> H3b(LLM-based Evolution:<br/>Generate New Prompt(s)<br/>using Operators & Feedback);
    H3b --> H4(Evaluate Candidate Prompt(s)<br/>(Call Critic on failing Task/Test<br/>with new prompt/old code? Or new code?)); %% Clarification needed: Does Evo evaluate prompt on OLD code + NEW prompt, or NEW code generated by NEW prompt? Assuming NEW code generated by NEW prompt for runtime self-healing.
    H4 --> H5(Update Runtime Population<br/>based on Evaluation Score);
    H5 --> H6(Select Best Prompt<br/>from Population);
    H6 --> L(Update Executor's Runtime Prompt); % This L node is the connection point
    end

    L(Updated Prompt) --> C; % Connects Self-Healing back to Executor
```

**Agent Roles:**

* **Planner Agent:** Receives the initial task description. Understands the request and provides a high-level structure or approach to the Executor. Its prompt is static per task instance.
* **Executor Agent:** Receives instructions from the Planner and a dynamic prompt from the Prompt Self-Healing module. Generates the Python code. Its prompt is the target of the dynamic modification.
* **Critic Agent:** Receives generated Python code. Evaluates correctness by running code, generating/running simple test cases, and analyzing results. Provides a structured failure report including error details and a quantifiable score (e.g., number of tests passed). This score serves as the $f_\mathcal{D}$ evaluation in the runtime EvoPrompt.
* **Prompt Modifier (Self-Healing Module):** This module is triggered by the Critic's failure detection. It implements the EvoPrompt algorithm. It maintains a small population of prompts during a task instance, uses LLM calls with defined "Evolution" operators to generate new prompt candidates based on the Critic's feedback, uses the Critic's evaluation function to score candidates on the *failing task instance*, updates its internal population, and selects the best prompt found within its limited iterations to pass back to the Executor.

## 5. Technical Specifications

* **Agent Orchestration Framework:** Crew AI is the preferred framework. Investigation is required to confirm its capability to manage agent states, dynamically modify an agent's prompt within a loop, and orchestrate calls between agents and the Prompt Modifier module. An alternative framework may be considered if Crew AI is unsuitable for this dynamic orchestration of the constrained EvoPrompt.
* **LLM Provider:**
    * Primary: Deepseek API. Its cost is beneficial for the iterative "Evolution" and "Evaluation" steps within the constrained EvoPrompt.
    * Alternative: OpenAI API. The implementation should abstract LLM calls.
* **Prompt Types:** All prompts (initial and modified) will be discrete text strings.
* **Failure Detection & Evaluation (Critic Implementation):**
    * The Critic receives the Python code as a string.
    * It executes the code within a secure sandbox environment, capturing `stdout` and `stderr`.
    * It analyzes `stderr` for syntax and runtime errors.
    * It automatically generates 1-2 simple, representative test cases (input/expected output) based on the original task description and code. This may involve an LLM call.
    * It runs the executed code with the generated test cases and compares actual output to expected output.
    * It produces a structured failure report including error details and test results. Crucially, it calculates a **quantifiable score** for the prompt/code pair, acting as the runtime $f_\mathcal{D}$ evaluation (e.g., number of generated tests passed, inverse of error severity).
* **Prompt Modification (Constrained EvoPrompt MVP):**
    * Algorithm 1 (EvoPrompt) will be adapted for runtime self-healing.
    * **Population ($P_t$)**: A small population of prompts will be maintained *per task instance* during the self-healing loop (e.g., size N=2 or 3). It might initially contain the failing prompt and potentially variants.
    * **Dev Set ($\mathcal{D}$) & Score ($f_\mathcal{D}$)**: The specific *failing task instance* serves as the development set. The Critic's evaluation score on this task instance (using the code generated by a given prompt) is the score function $f_\mathcal{D}$.
    * **Iterations ($T$)**: A very small, fixed number of evolutionary iterations ($T=1$ or $T=2$) will run *each time* the Self-Healing module is triggered by the Critic.
    * **Selection:** Simple selection rules (e.g., select the prompt with the best score from the current runtime population, or the current failing prompt and the best historical prompt).
    * **Evolution ($Evo$)**: An LLM call will be used to perform the evolutionary operation(s). The input will be the selected parent prompt(s) and the Critic's failure feedback. The LLM will be prompted to generate a *new discrete prompt* based on this information. Example operators (defined via prompting the LLM): "Modify this prompt to fix error X", "Rewrite this prompt to try a different approach given test failure Y", "Combine elements of prompt A and prompt B to address feedback Z". The goal is to make holistic prompt improvements that enhance the agent's overall capability rather than just fixing the immediate error. For instance, a prompt might evolve as follows:
        *   **Original Prompt:** "You are a meticulous Python programmer. Write code according to the spec. Output only code."
        *   **Iteration 1 - Evolved Prompt:** "You are a software engineer writing production-ready Python code. Avoid relying on high-level shortcuts; focus on correctness and edge-case handling."
        *   **Iteration 2 - Evolved Prompt:** "You are a senior software engineer writing production-ready Python code that is explicit, handles inputs defensively, and includes minimal inline testing to ensure correctness. Avoid relying on high-level shortcuts; Return only valid Python code."
    * **Evaluation:** The newly generated candidate prompt($p_i'$) will be evaluated by triggering the Critic Evaluation process *using this new prompt* on the *original failing task instance*. The Critic will run the *new code* generated by the Executor using $p_i'$ and provide a score $s_i' = f_\mathcal{D}(p_i')$.
    * **Update:** The candidate prompt($p_i'$) and its score($s_i'$) will be added to the small runtime population. The population might then be pruned to maintain the size $N$ (e.g., keep the $N$ prompts with the highest scores).
    * **Return:** After $T$ iterations of this mini-Evo process, the prompt with the highest score in the current runtime population is selected and returned to update the Executor's prompt for the main system loop.
* **Self-Healing Loop Limit:** The *main system loop* (Executor -> Critic -> Self-Heal) should run for a maximum of 2-3 iterations per task instance to manage complexity, regardless of the number of internal EvoPrompt iterations ($T$).

## 6. Success Metrics

The success of the MVP will be evaluated based on:

* **Loop Execution Rate:** Percentage of test cases where the full Planner -> Executor -> Critic -> Self-Heal (Constrained EvoPrompt) -> Executor loop successfully executes without crashing or getting stuck (e.g., target > 90%).
* **Prompt Modification Effectiveness:** Demonstrate that the Prompt Modifier generates distinct, failure-informed prompts using the LLM-based evolutionary operators and that the evaluation step using the Critic is functional.
* **Quantifiable Improvement:** For test cases where the initial Executor output fails the Critic's generated tests, measure the percentage of these cases that show an *increased* test pass rate (from the Critic's generated suite) in the subsequent self-healing attempt(s) driven by the EvoPrompt selected prompt. (e.g., target > 50% of initially failed cases show improvement).
* **Integration Success:** Successful deployment and execution using the chosen framework and LLM provider, demonstrating the orchestration of agents and the internal EvoPrompt evaluation calls.

## 7. Out of Scope

The following functionalities and complexities from the full research proposal are explicitly out of scope for this MVP:

* Optimizing the initial *population* $P_0$ for the EvoPrompt process itself; the runtime population will be initialized based on the first failing prompt.
* Running the full EvoPrompt algorithm offline or over a large, static development set $\mathcal{D}$. The algorithm is constrained and run *dynamically* upon failure using the specific task instance for evaluation.
* Implementing complex, multi-step evolutionary operators or crossover operations between disparate prompts.
* Utilizing or experimenting with continuous (soft) prompts.
* Handling multi-file Python projects or complex project structures.
* Implementing sophisticated code optimization checks in the Critic (beyond aiming for correct and simply structured code).
* Developing a comprehensive suite of predefined unit tests for evaluation; relying primarily on the Critic's generated tests and runtime errors.
* Implementing the full optimization loop for recovery-aware cumulative reward as a training signal for the prompt modification process itself (the process is rule-based/heuristic selection for MVP).
* Handling ambiguous or underspecified task descriptions.

## 8. Future Work

Potential future work beyond the MVP includes:

* Implementing a full offline EvoPrompt run to optimize the *initial* prompts ($P_0$) for the system.
* Developing more sophisticated LLM-based evolutionary operators.
* Exploring different population management and selection strategies for the runtime EvoPrompt.
* Incorporating continuous prompt representations.
* Expanding the task scope to more complex coding challenges and languages.
* Developing more advanced Critic capabilities (e.g., static analysis, performance checks).
* Exploring different agent architectures and roles.
* Implementing a learned policy ($\pi$) that *selects* which type of prompt modification (including potentially triggering a more complex Evo process or other strategies) is best given a failure state.
* Implementing the full optimization loop for recovery-aware cumulative reward across multiple tasks.

---

# Development Task List: Self-Healing Agentic AI MVP

This is a proposed task list for developing the Self-Healing Agentic AI MVP, incorporating the Constrained EvoPrompt approach.

## Phase 1: Core Infrastructure & Basic Agent Workflow Setup

* [x] Task 1.1: **Orchestration Framework Setup & Verification:**
    *   Select and set up the primary Agent Orchestration Framework (e.g., Crew AI).
    *   Verify its core capabilities for this MVP:
        *   Managing state for Planner, Executor, and Critic agents.
        *   Facilitating sequential and conditional inter-agent communication.
        *   Dynamically updating the Executor agent's prompt at runtime *within* a task execution loop.
        *   Handling return values and structured data between agents.
* [x] Task 1.2: **LLM Provider Integration & Abstraction:**
    *   Select and configure the primary LLM Provider API (e.g., Deepseek).
    *   Implement an LLM service abstraction layer/class to encapsulate API calls (e.g., `LLMService(provider='deepseek')`). This will allow for easier switching or adding alternative providers (e.g., OpenAI) later.
    *   Ensure the abstraction supports passing system messages, user messages, and receiving structured (e.g., JSON) or text outputs.
    *   Write unit tests for the `LLMService` abstraction layer to verify its core functionalities (e.g., request formatting, basic response handling, error propagation) before full integration.
* [x] Task 1.3: **Initial Agent Prompt Definition (Version 1):**
    *   Define the initial, static (for now) system and task-level prompts for the Planner Agent (to understand goal and outline structure).
    *   Define the initial (V1) system prompt for the Executor Agent (this will be the first prompt targeted by EvoPrompt); keep this prompt relativly simple.
    *   Define the initial system prompt for the Critic Agent (guiding its code analysis and test generation behavior).
* [x] Task 1.4: **Implement Basic Sequential Agent Workflow (No Self-Healing):**
    *   Implement the direct flow: User Request -> Planner -> Executor -> Critic (initial evaluation).
    *   Ensure data (task description, code structure, generated code) is passed correctly between agents.
    *   Develop integration tests for this basic workflow, verifying agent communication and data handoff, ideally before or alongside implementing the full agent logic.
* [x] Task 1.5: **Placeholder Critic Logic for Loop Triggering:**
    *   In the Critic agent, implement placeholder logic to receive code from the Executor.
    *   This placeholder should deterministically return:
        *   A dummy "failure" report (with a placeholder score indicating failure) for predefined test inputs/tasks.
        *   A dummy "success" report (with a placeholder score indicating success) for other test inputs/tasks.
    *   This will enable testing the basic control flow into and out of the (yet to be implemented) self-healing loop.
    *   Write unit tests for this placeholder logic to ensure it returns the expected dummy reports under different conditions.
* [x] Task 1.6: **End-to-End Basic Flow Test:**
    *   Develop 1-2 extremely simple Python generation tasks.
    *   Test the flow from Planner -> Executor -> Critic (with placeholder logic) to ensure:
        *   Agents communicate in the correct sequence.
        *   The placeholder "failure" and "success" conditions can be triggered as expected, setting the stage for the self-healing loop.

## Phase 2: Critic Implementation (Failure Detection & Evaluation $f_\mathcal{D}$)

* [x] Task 2.1: **Secure Code Execution Environment:**
    *   Implement a secure sandbox environment within the Critic agent to execute arbitrary Python code strings received from the Executor.
    *   Consider using `RestrictedPython`, `exec` with carefully controlled `globals` and `locals`, or a lightweight containerization approach if feasible for the MVP.
    *   Ensure network access and filesystem access are appropriately restricted from within this sandbox.
    *   Develop unit tests to verify the security constraints and correct execution of valid/invalid code snippets within the sandbox before implementing more complex Critic logic.
* [x] Task 2.2: **Error Capture and Structured Parsing:**
    *   Implement robust capture of `stdout` and `stderr` from the sandboxed code execution.
    *   Parse `stderr` to identify and categorize Python errors (e.g., `SyntaxError`, `NameError`, `TypeError`, `IndexError`, `ZeroDivisionError`).
    *   Extract key error information: type, message, and a simplified traceback.
    *   Write unit tests for the error parsing logic, covering various error types and formats, before integrating it into the Critic.
* [x] Task 2.3: **Dynamic Test Case Generation (Input/Output Oracle):**
    *   Implement logic within the Critic, using an LLM call (via the `LLMService`), to generate test cases representing the given Python code.
    *   Input to this generation process should include the original task description (from Planner/User) and the generated code itself.
    *   Each test case should consist of sample input(s) and expected output(s) or a condition to verify.
    *   While the LLM's generation is stochastic, write tests for the surrounding logic: formatting inputs for the LLM and parsing its output into structured test cases.
* [x] Task 2.4: **Test Case Execution and Result Comparison:**
    *   Implement logic to run the Executor's code (within the sandbox) against each dynamically generated test case.
    *   Capture the actual output produced by the code for each test case input.
    *   Compare the actual output against the expected output, handling potential exceptions during test execution.
    *   Employ TDD: write tests defining expected outcomes for various code behaviors (correct output, incorrect output, exceptions during test) before implementing the comparison logic.
* [x] Task 2.5: **Comprehensive Result Analysis:**
    *   Consolidate findings from direct code execution (syntax/runtime errors) and test case execution (pass/fail status).
    *   Develop logic to differentiate between critical errors (e.g., syntax errors preventing execution) and logical errors (e.g., failing test cases but code runs).
    *   Write unit tests for this consolidation and differentiation logic, covering various combinations of error types and test results.
* [x] Task 2.6: **Implement Quantifiable Scoring Function ($f_\mathcal{D}$):**
    *   **This is a critical task THIS HAS TO BE DONE METICULOUSLY.** Design and implement the scoring function within the Critic that produces a quantifiable score for a (prompt, generated code) pair.
    *   The score should reflect the code's correctness and serve as the fitness function for EvoPrompt. For example:
        *   A normalized score (e.g., 0.0 for complete failure, 1.0 for all tests passed and no errors) all scores should be between 0 and 1.
        *   Consider penalizing based on error severity (e.g., syntax error = 0, runtime error = low score, X% of tests passed = proportional score).
        *   Ensure the score is granular enough to guide the evolutionary process (i.e., minor improvements in code should lead to score improvements).
    *   Develop this scoring function using TDD: define test cases with specific code outputs/error conditions and their expected scores before writing the scoring logic.
* [x] Task 2.7: **Structured Feedback Report Generation:**
    *   Integrate all analysis into a structured failure/success report to be returned by the Critic.
    *   This report must include:
        *   Overall status (e.g., `SUCCESS`, `FAILURE_SYNTAX`, `FAILURE_RUNTIME`, `FAILURE_LOGIC`).
        *   The quantifiable score ($f_\mathcal{D}$).
        *   Details of any execution errors (type, message, simplified traceback).
        *   Results of each generated test case (input, expected output, actual output, pass/fail).
        *   Concise summary of issues to inform the Prompt Modifier.

## Phase 3: Constrained EvoPrompt Self-Healing Implementation

* [x] Task 3.1: **Self-Healing Trigger Logic:**
    * Implement logic (likely in the orchestrator or a managing agent) to check the Critic's structured report.
    * Trigger the Prompt Modifier (Self-Healing Module) if the Critic's report indicates a failure (e.g., score below a success threshold, or a specific failure status).
    * Write unit tests to verify this trigger logic correctly identifies failure conditions based on various sample Critic reports.
* [x] Task 3.2: **Prompt Modifier Module/Class Structure:**
    * Design and implement the `PromptModifier` module (likely as a class) that encapsulates the Constrained EvoPrompt logic.
    * This module will be stateful *per task instance*, managing its internal prompt population and scores.
    * Define initial unit tests for the `PromptModifier` class structure and its core methods (even if initially failing) before full implementation of EvoPrompt steps.
* [x] Task 3.3: **Runtime Prompt Population Management (Initialization & Storage):**
    * Within `PromptModifier`, implement logic to maintain a small runtime population of (prompt string, score, age/iteration_created) tuples. Population size $N$ (e.g., 2-3 as per PRD).
    * Initialize this population upon the *first* trigger for a task instance: typically with the current failing Executor prompt and its score from the Critic. Consider if simple variants can be added immediately.
    * Use TDD: write tests for population initialization, adding new prompts, and retrieving prompts before implementing the logic.
* [x] Task 3.4: **Parent Prompt Selection Strategy (EvoPrompt - Selection):**
    * Implement the selection mechanism for choosing parent prompts from the current runtime population.
    * For MVP, this could be: selecting the current failing prompt and the historically best-scoring prompt in the population, or just the current failing prompt if it's the only one or the first iteration of self-healing.
    * Develop unit tests for the selection strategy with various population states to ensure correct parent(s) are chosen, before implementing the selection logic.
* [x] Task 3.5: **LLM-based Prompt Evolution (EvoPrompt - Evolution $Evo$):**
    * **This is a critical task.** Implement the "Evolution" step using the `LLMService`.
    * The LLM call should be prompted with:
        * The selected parent prompt(s).
        * The structured failure report and score from the Critic (for the code generated by the parent prompt).
        * Instructions to generate 1-2 *new candidate holistic system prompts* for the Executor, aiming to address the reported failures and improve the score, drawing on the PRD's examples of holistic prompt evolution.
    * Ensure the LLM is guided to produce complete, new system prompts, not just edits or suggestions.
    * Test the surrounding logic: formatting inputs to the LLM for evolution and parsing its output, even if the LLM's output itself is non-deterministic.
* [x] Task 3.6: **Candidate Prompt Evaluation (EvoPrompt - Evaluation $f_\mathcal{D}$ via Orchestration):**
    * **This is a critical task.** For each new candidate prompt generated in Task 3.5:
        * The `PromptModifier` must orchestrate a sub-loop: provide the candidate prompt to the Executor agent.
        * The Executor generates new code based *only* on this candidate prompt and the original Planner instructions.
        * The `PromptModifier` then passes this new code to the Critic agent for evaluation on the *original failing task instance*.
        * The Critic returns its structured report and score ($s_i' = f_\mathcal{D}(p_i')$) for this candidate prompt's output.
        * While this is an orchestration task, ensure the interactions (e.g., `PromptModifier` calling Executor, then Critic) can be integration-tested, perhaps with mocks for the agents initially.
* [x] Task 3.7: **Population Update and Pruning (EvoPrompt - Update):**
    * **This is a critical task.** Add the newly evaluated candidate prompt(s) and their scores ($p_i', s_i'$) to the `PromptModifier`'s runtime population.
    * Implement population pruning to maintain size $N$: e.g., keep the $N$ prompts with the highest scores (elitism). Consider tie-breaking rules (e.g., favor newer prompts or prompts that led to score improvement).
    * Write unit tests for the update and pruning logic, covering scenarios like adding to a full population, handling ties, etc., prior to implementation.
* [x] Task 3.8: **Best Prompt Selection for Next System Iteration (EvoPrompt - Return):**
    * After the fixed number of internal EvoPrompt iterations ($T=1$ or $T=2$, as per PRD) within the `PromptModifier` for a single self-healing trigger:
    * Select the prompt with the absolute highest score from the current runtime population in the `PromptModifier`.
    * Use TDD: define test cases for selecting the best prompt from various sample populations before implementing the selection logic.
* [x] Task 3.9: **Dynamic Executor Prompt Update:**
    * Implement the mechanism for the `PromptModifier` (or orchestrator) to pass the selected best prompt (from Task 3.8) back to the Executor agent.
    * The Executor agent must use this newly supplied prompt for its subsequent attempt on the task within the main system loop.
* [x] Task 3.10: **Main System Self-Healing Loop Control:**
    * Implement the overarching control logic (likely in the orchestrator) for the main system loop: Executor -> Critic -> (if failure) PromptModifier -> Executor.
    * This loop should run for a maximum of 2-3 iterations *per task instance* (as per PRD Section 5), regardless of the number of internal EvoPrompt iterations ($T$) within the PromptModifier on each trigger.
    * Develop tests for this loop control logic, ensuring it correctly counts iterations and terminates as expected under different conditions (success, max iterations reached).

## Phase 4: Comprehensive Testing and MVP Evaluation

* [x] Task 4.1: **Develop Diverse Test Task Suite:**
    *   Create a small suite of 5-10 diverse Python code generation/debugging tasks.
    *   Tasks should vary in complexity (e.g., single function, simple script) and be designed to trigger different initial failure modes (e.g., syntax errors, common runtime errors like `TypeError` or `IndexError`, simple logical flaws detectable by basic tests).
    *   For each task, define the high-level goal/description that would be provided to the Planner agent.
* [x] Task 4.2: **Automated Evaluation Harness Development:**
    *   Develop an evaluation script or harness to automate running the entire MVP system (Planner -> Executor -> Critic -> Self-Heal loop) on each test task from the suite.
    *   The harness should handle initializing agents, triggering the first task, and letting the self-healing loop run to completion (either success or max iterations reached).
* [ ] Task 4.3: **Detailed Per-Iteration Logging for Analysis:**
    *   Instrument the system (or the evaluation harness) to log detailed information for each main system iteration and each internal EvoPrompt iteration:
        *   **Task Instance ID.**
        *   **Main System Iteration Number.**
        *   **Planner Agent:** Input task, output structure/plan.
        *   **Executor Agent:** Current system prompt used, generated code.
        *   **Critic Agent:** Full structured report (input code, errors, generated tests, test results, score $f_\mathcal{D}$).
        *   **PromptModifier (if triggered):**
            *   Input (failing prompt, Critic report).
            *   Internal EvoPrompt Iteration Number ($t=1..T$).
            *   Selected parent prompt(s) for evolution.
            *   Generated candidate prompt(s) by LLM-Evo step.
            *   Score of each candidate prompt after its evaluation run (Executor new code -> Critic new score).
            *   State of the runtime population after update.
            *   Selected best prompt returned to Executor.
        *   **Final Outcome:** Overall task success/failure, total iterations.
* [ ] Task 4.4: **Execute Test Suite and Collect Data:**
    *   Run the full evaluation harness across all defined test tasks.
    *   Collect and store the detailed logs in a structured format (e.g., JSON lines, CSV per task) for analysis.
* [ ] Task 4.5: **Analyze Results Against PRD Success Metrics:**
    *   Process the collected logs to calculate the success metrics defined in PRD Section 6:
        *   **Loop Execution Rate:** Percentage of tasks where the full self-healing loop executes as intended without system crashes.
        *   **Prompt Modification Effectiveness:** Qualitative analysis of generated prompt variants (are they distinct and failure-informed?). Quantitative measure of how often EvoPrompt produces a prompt different from its input.
        *   **Quantifiable Improvement:** For tasks that initially failed, percentage of cases showing an *increased* Critic score in subsequent self-healing attempts. Measure average score lift.
        *   **Integration Success:** Confirmation of end-to-end functionality with chosen framework/LLM.
* [ ] Task 4.6: **Identify Systemic Failure Modes and Bottlenecks:**
    *   Analyze logs for recurring issues or patterns in failures:
        *   Critic failing to detect obvious errors or generating poor tests.
        *   EvoPrompt generating nonsensical or unhelpful prompt variations.
        *   The system getting stuck in repetitive prompt generation cycles without score improvement.
        *   Orchestration framework limitations or bugs encountered.
        *   LLM issues (e.g., refusing to modify prompts, generating off-topic content).

## Phase 5: Iterative Refinement and MVP Documentation

* [ ] Task 5.1: **Address Critical System Bugs and Stability Issues:**
    *   Prioritize and fix any critical bugs identified in Phase 4 that prevent the reliable execution of the self-healing loop or significantly skew results (e.g., crashes, consistent failure of Critic to parse code, major issues in orchestration).
    *   For each bug, write a regression test that reproduces the bug *before* fixing it. Ensure the test passes after the fix.
* [ ] Task 5.2: **Targeted Refinement of Key Modules based on Analysis:**
    *   **EvoPrompt LLM Evolution Step:** Based on analysis of generated prompts (Task 4.5, 4.6), refine the prompt templates and instructions provided to the LLM during the "Evolution" step (Task 3.5). This might involve adjusting the meta-prompt to better guide the LLM towards producing effective, holistic prompt variations.
    *   **Critic Scoring Function ($f_\mathcal{D}$):** If analysis shows the scoring function (Task 2.6) is not granular enough, or rewards/penalizes inappropriately, revise its logic to better reflect true code quality and guide EvoPrompt more effectively.
    *   **Critic Test Case Generation:** If generated tests are consistently poor or irrelevant (Task 4.6), refine the logic or LLM prompting for test case generation (Task 2.3).
    *   Any code changes made during refinement should be accompanied by new or updated unit/integration tests verifying the intended improvement and guarding against regressions.
* [ ] Task 5.3: **Tune EvoPrompt Parameters (If Necessary and Feasible):**
    *   Based on results, consider minor adjustments to Constrained EvoPrompt parameters if clear deficiencies are noted:
        *   Selection strategy for parent prompts (Task 3.4).
        *   Population update/pruning rules (Task 3.7).
        *   (Note: Major changes to EvoPrompt algorithm itself are out of scope for MVP, focus on tuning existing mechanisms).
* [ ] Task 5.4: **Final MVP Documentation and Knowledge Capture:**
    *   Document the final state of the MVP, including:
        *   Key design decisions and rationale for Planner, Executor, Critic, and PromptModifier agents/modules.
        *   Finalized prompt templates used for LLM-based evolution and Critic test generation.
        *   Observed behavior of the Constrained EvoPrompt in runtime self-healing scenarios.
        *   Lessons learned regarding the effectiveness of the approach, LLM behavior, and orchestration challenges.
    *   Summarize findings and provide concrete recommendations for future work (as outlined in PRD Section 8).

---
Please check of the task in Task List as you perform them. Remeber perform at your best and you will receive a special reward.