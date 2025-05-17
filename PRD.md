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
* **Task Complexity:** Generating and debugging single functions or small scripts (less than ~50 lines).
* **Error Triggering:** The system should be capable of detecting and being *triggered* by common Python errors within the MVP task scope, including syntax errors, basic runtime errors (e.g., division by zero, index errors), and failures against simple logical tests.
* **Self-Healing Mechanism:** Implementation of a *discrete* (text-based) prompt modification strategy based on a *constrained, runtime application* of an evolutionary optimization algorithm (EvoPrompt).
* **Evaluation:** The Critic agent will perform implicit judgment augmented by generating and running simple test cases to provide quantifiable feedback used by the self-healing mechanism.

## 2. Goals

The specific, measurable goals for the Self-Healing Agentic AI MVP are:

* Successfully demonstrate the execution of the full agentic self-healing loop (Planner -> Executor -> Critic -> Failure Detection -> Prompt Modification via Constrained EvoPrompt -> Loop back to Executor) on a predefined set of small Python coding tasks.
* Implement the Critic agent's capability to run Python code, capture errors (syntax, runtime), and generate/run simple test cases against the code, providing structured feedback and a quantifiable score.
* Implement a constrained version of the EvoPrompt algorithm within the Prompt Modifier module to generate a revised discrete prompt for the Executor based on the Critic's failure feedback and evaluation score.
* Show quantifiable improvement in the generated code's correctness (measured by the Critic's generated test pass rate score) after one or more self-healing iterations in a majority of test cases where the initial attempt failed.
* Successfully integrate the chosen agent orchestration framework (Crew AI or alternative) and selected LLM provider (Groq/Hugging Face API) to support the dynamic prompt modification and LLM-based "Evolution" step required by the EvoPrompt process.

## 3. User Stories (Implicit System Behaviors)

* As the system, when given a request to write a small Python function, I should generate the initial code.
* As the system, when my initial code fails during execution or against a test, I should detect the failure and receive structured feedback from my evaluation component.
* As the system, upon receiving failure feedback, I should use this information within a prompt optimization process to generate better instructions for my code-writing component.
* As the system, I should attempt to fix the code based on the newly optimized instruction.
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
* **Prompt Modifier (Self-Healing Module):** This module is triggered by the Critic's failure detection. It implements a constrained version of the EvoPrompt algorithm. It maintains a small population of prompts during a task instance, uses LLM calls with defined "Evolution" operators to generate new prompt candidates based on the Critic's feedback, uses the Critic's evaluation function to score candidates on the *failing task instance*, updates its internal population, and selects the best prompt found within its limited iterations to pass back to the Executor.

## 5. Technical Specifications

* **Agent Orchestration Framework:** Crew AI is the preferred framework. Investigation is required to confirm its capability to manage agent states, dynamically modify an agent's prompt within a loop, and orchestrate calls between agents and the Prompt Modifier module. An alternative framework may be considered if Crew AI is unsuitable for this dynamic orchestration of the constrained EvoPrompt.
* **LLM Provider:**
    * Primary: Groq API. Its speed is beneficial for the iterative "Evolution" and "Evaluation" steps within the constrained EvoPrompt.
    * Alternative: Hugging Face API or a local serving method. The implementation should abstract LLM calls.
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
    * **Evolution ($Evo$)**: An LLM call will be used to perform the evolutionary operation(s). The input will be the selected parent prompt(s) and the Critic's failure feedback. The LLM will be prompted to generate a *new discrete prompt* based on this information. Example operators (defined via prompting the LLM): "Modify this prompt to fix error X", "Rewrite this prompt to try a different approach given test failure Y", "Combine elements of prompt A and prompt B to address feedback Z".
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

## Phase 1: Setup and Basic Workflow

* [] Task 1.1: Select and set up the Agent Orchestration Framework (Crew AI or alternative). Verify its ability to manage agent state and dynamically update prompts within a loop.
* [] Task 1.2: Select and configure the LLM Provider API (Deepseek). Implement abstraction for LLM calls.
* [] Task 1.3: Define the initial prompts for the Planner, Executor, and Critic agents.
* [] Task 1.4: Implement the basic sequential workflow: Planner -> Executor -> Critic.
* [] Task 1.5: Create placeholder logic in the Critic to receive code and return dummy "failure" feedback to trigger the loop.
* [] Task 1.6: Test the basic flow with a simple task to ensure agents communicate and execute in sequence, hitting the placeholder failure.

## Phase 2: Critic Implementation (Failure Detection & Evaluation $f_\mathcal{D}$)

* [] Task 2.1: Implement secure Python code execution capability within the Critic (run code string in a sandbox).
* [] Task 2.2: Implement capture and analysis of `stdout` and `stderr` for syntax and runtime errors.
* [] Task 2.3: Implement logic (potentially using an LLM call) to generate 1-2 simple, representative test cases (input/expected output) based on the original task description and code.
* [] Task 2.4: Implement logic to run the executed code against the generated test cases and capture results.
* [] Task 2.5: Implement logic to analyze test results (pass/fail) and combine with error analysis.
* [] Task 2.6: **Implement the scoring function ($f_\mathcal{D}$) for the Critic.** This function takes the task, code, and results and returns a quantifiable score (e.g., number of tests passed, a composite score considering error type). This score will be used by EvoPrompt.
* [] Task 2.7: Integrate the failure detection and scoring logic into the Critic's main process, returning a structured failure report including the score.

## Phase 3: Constrained EvoPrompt Self-Healing

* [] Task 3.1: Implement the logic to check the Critic's output for a detected failure and trigger the Self-Healing module.
* [] Task 3.2: Implement the Prompt Modifier module/function which orchestrates the constrained EvoPrompt process.
* [] Task 3.3: Implement logic within the Prompt Modifier to maintain a small runtime population of prompts and their scores for the current task instance. Initialize it upon the first failure (e.g., with the failing prompt).
* [] Task 3.4: Implement the "Selection" step for the runtime population (e.g., select the current best prompt and the failing prompt as parents if different).
* [] Task 3.5: **Implement the "Evolution" step ($Evo$) using an LLM call.** This involves prompting the LLM with selected parent prompt(s) and the Critic's failure report/feedback to generate 1-2 new candidate prompt strings according to predefined simple operators (e.g., "fix error X", "try different approach").
* [] Task 3.6: **Implement the "Evaluation" step within the Prompt Modifier.** For each candidate prompt generated in Task 3.5, call the Executor agent *using this candidate prompt* to generate new code, then call the Critic agent to evaluate this new code on the *failing task instance*. Receive the score ($s_i'$) from the Critic.
* [] Task 3.7: **Implement the "Update" step.** Add the candidate prompt(s) and their scores to the runtime population. Implement population management (e.g., keep the top N prompts by score).
* [] Task 3.8: Implement the "Return/Selection" step after the limited internal EvoPrompt iterations ($T$). Select the prompt with the highest score from the current runtime population.
* [] Task 3.9: Implement the mechanism to dynamically update the Executor agent's prompt with the selected best prompt for the next iteration of the main system loop.
* [] Task 3.10: Implement the main system loop control to re-run the Executor -> Critic -> Self-Heal cycle upon failure, for a maximum number of outer iterations (e.g., 5 attempts total per task).

## Phase 4: Testing and Evaluation

* [ ] Task 4.1: Create a small set of diverse test tasks for Python code generation/debugging, including cases expected to initially fail with different error types (syntax, runtime, simple logic). Aim for 5-10 distinct tasks.
* [ ] Task 4.2: Develop an evaluation script or process to run the MVP system on each test task instance from start to finish (including all self-healing attempts).
* [ ] Task 4.3: Modify the system or evaluation script to log/track the initial prompt, all subsequent prompts generated by EvoPrompt, the Critic's full failure report and score for each attempt, and the final outcome (success/failure) for each task instance.
* [ ] Task 4.4: Run the test suite and collect detailed results logs.
* [ ] Task 4.5: Analyze the logs to calculate success metrics (Loop Execution Rate, ability of EvoPrompt to generate variants, ability of Critic to score, Quantifiable Improvement in score after EvoPrompt step) against the goals defined in the PRD.
* [ ] Task 4.6: Identify common failure modes of the MVP system itself (e.g., Critic fails to detect error, EvoPrompt generates poor candidates, the loop gets stuck, framework issues).

## Phase 5: Refinement (Based on MVP Findings)

* [ ] Task 5.1: Address critical bugs identified during testing.
* [ ] Task 5.2: Analyze the EvoPrompt behavior and Critic scores. Refine prompt templates used for the LLM "Evolution" step (Task 3.5) and the Critic's scoring function (Task 2.6) if analysis shows they are ineffective.
* [ ] Task 5.3: Refine the selection mechanism within EvoPrompt (Task 3.4 & 3.8) or population management (Task 3.3 & 3.7) based on results.
* [ ] Task 5.4: Document findings, lessons learned about using constrained EvoPrompt for runtime self-healing, and recommendations for future work.

---
Please check of the task in Task List as you perform them. Remeber perform at your best and you will receive a special reward.