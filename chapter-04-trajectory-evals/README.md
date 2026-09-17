# Chapter 4 -- Trajectory Evals
### Lab: Trajectory Evaluation Engine

> Companion to Chapter 4 of *Agentic Evals System Design*.

An IT helpdesk agent diagnoses a ticket through a live tool-calling loop: the model chooses each
diagnostic action, a scenario-consistent simulated environment answers, and a milestone-based
trajectory evaluator scores the recorded path.

## Files

```text
chapter-04-trajectory-evals/
├── tools.py       # ITEnvironment (vpn / disk / sso state machines), SCENARIOS (milestones, optimal steps), TOOL_SCHEMAS
├── engine.py      # ITHelpdeskAgent: provider.chat(..., tools=...) loop, max 8 steps
├── evaluator.py   # ITTrajectoryEvaluator -> shared/evaluators/trajectory.py
└── tests/
    └── test_trajectory.py   # 4 tests
```

UI page: `pages/4_Ch4_Trajectory_Evals.py`.

## How grading works (`shared/evaluators/trajectory.py`)

* Every step carries a kind recorded by the tool: `success`, `finding` (the probe worked and found a
  problem), or `tool_failure`. Findings are not failures.
* **Progress** -- share of scenario milestones reached. Milestones are resolution actions, never diagnostic ones.
* **Recovery** -- a failure counts as recovered only if a later step reaches a milestone not reached before it.
* **Wasted repeats / loop penalty** -- same action and arguments **and** an observation already seen. Polling with a changing observation is free.
* **Score** -- `100 * (0.5*progress + 0.3*efficiency + 0.2*(1 - min(1, wasted/2)))`, gated to 0 when the outcome failed.

Outcome is decided independently of the agent's narration: every milestone action must have succeeded.

## Running the lab

```bash
streamlit run Home.py   # open "Chapter 4" in the sidebar
```

Pick a ticket (VPN, disk, SSO), optionally inject a failing `compress_logs` loop in the disk
scenario, and click **Run live diagnosis**. The page shows the trajectory metrics with the
evaluator's reasoning, every step with its kind, and the agent's final answer.

## Tests

```bash
MOCK_LLM=1 pytest chapter-04-trajectory-evals/tests/test_trajectory.py -v
```
