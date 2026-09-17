# Chapter 5 -- LLM-as-a-Judge
### Lab: Build a Local LLM Judge

> Companion to Chapter 5 of *Agentic Evals System Design*.

A judge that fails closed, a second judge whose disagreement routes pairs to human review, a
verbosity-bias probe, and a calibration report against human labels that refuses to overstate
what five labels can show.

## Files

```text
chapter-05-llm-judge/
├── judge.py         # MultiJudgeSystem: primary + alternative judge, disagreement routing, probe_verbosity_bias
├── calibration.py   # compute_calibration_metrics: agreement, Cohen's kappa + bootstrap CI, FP/FN rates, Pearson, under-powered warning
└── tests/
    └── test_judge.py   # 4 tests
```

Shared judge: `shared/evaluators/llm_judge.py`. UI page: `pages/5_Ch5_LLM_Judge.py`.
Dataset: `shared/datasets/data/human_benchmark_judge.jsonl` (5 items).

## How the judge works

* Rubric descriptions **and** scoring anchors are rendered into the prompt.
* The reply must validate against `JudgeOutput` (all five dimensions, 0-5, a reasoning sentence).
* A reply that fails validation gets one corrective retry; a second failure **abstains** (`passed=None`).
* The overall score is a weighted mean computed in code (safety weight 1.5), never read from the model.
* A hard floor fails the verdict when safety < 4.0, whatever the overall score.

## Running the lab

```bash
ollama pull qwen3:1.7b && ollama pull llama3.2:1b
streamlit run Home.py   # open "Chapter 5" in the sidebar
```

* **Judge a Response** -- edit the prompt, response, and optional context; click **Judge it**. Both
  judges' scores (or an ABSTAINED warning), a human-review banner when they disagree, and optionally
  the verbosity-bias probe.
* **Calibration Report** -- scores the benchmark with the primary judge and shows the metrics with
  the small-sample warning.

## Tests

```bash
MOCK_LLM=1 pytest chapter-05-llm-judge/tests/test_judge.py -v
```
