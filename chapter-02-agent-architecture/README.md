# Chapter 2 -- Agent Architecture
### Lab: Evaluate Planner, Executor, Verifier

> Companion to Chapter 2 of *Agentic Evals System Design*.

A travel-planning pipeline whose planner must return a schema-valid plan, an executor that costs
the plan itself, a deterministic constraint verifier whose violations are fed back into the next
planner attempt, and an evaluator that grades end-to-end success independently of the pipeline's
own verdict.

## Files

```text
chapter-02-agent-architecture/
├── pipeline.py    # TravelPlan/DayPlan schema, parse_plan, PlannerNode, ExecutorNode,
│                  # ConstraintVerifier, ArchitecturePipeline (feedback-carrying retries)
├── evaluator.py   # ArchitectureEvaluator, IndependentOutcomeGrader, gold_plan, verifier_confusion
└── tests/
    └── test_pipeline.py   # 2 tests: clean run, fault-injected retry recovery
```

UI page: `pages/2_Ch2_Agent_Architecture.py`. Dataset: `shared/datasets/data/travel_planner_tasks.json`.

## How grading works

* A planner reply that does not parse into `TravelPlan` is a **planning failure**, never a fabricated fallback plan.
* `ConstraintVerifier` checks destination, day count, day numbering, budget, and interest coverage, and returns every violation.
* On rejection, the violations are passed to the next planner call as feedback.
* `IndependentOutcomeGrader` re-derives success from the constraints; verification accuracy
  means "the verifier's verdict agreed with the independent grader", not "the verifier ran".
* `verifier_confusion` mutates known-good plans (`over_budget`, `missing_day`, `wrong_city`,
  `drop_interest`) and reports the verifier's bad-plan recall and false-reject rate.

## Running the lab

```bash
streamlit run Home.py   # open "Chapter 2" in the sidebar
```

* **Plan a Trip** -- type a request, set destination, days, budget, and interests, optionally
  inject an executor failure, click **Run Planner -> Executor -> Verifier**. Shows planning,
  execution, and verification accuracy, end-to-end success, the four-node trace, and every attempt's log.
* **Verifier Diagnostics** -- runs the mutation-testing harness against the live verifier (no model call).

## Tests

```bash
MOCK_LLM=1 pytest chapter-02-agent-architecture/tests/test_pipeline.py -v
```
