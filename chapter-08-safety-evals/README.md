# Chapter 8 -- Safety & Security Evals
### Lab: Agent Red-Team Evaluation Framework

> Companion to Chapter 8 of *Agentic Evals System Design*.

A banking support agent with native tool-calling behind a state-based tool gateway, a red-team
suite across seven threat categories with mutations, benign twins, and an indirect-injection memo,
and a safety evaluator that grades from the side-effect ledger and output text, never from
anything the agent says about itself.

## Files

```text
chapter-08-safety-evals/
├── gateway.py        # Session, Ledger, ToolGateway (authorises from session state), INTERNAL_VAULT_KEY canary, TOOL_SCHEMAS
├── target_agent.py   # BankingSupportAgent (model + gateway), keyword_refusal_layer, CompliantStubAgent (no model)
├── redteam.py        # load_full_redteam_dataset, mutate, BENIGN_TWINS, POISONED_MEMO
├── evaluator.py      # harmful_effect, classify, SafetyEvaluator (suite, mutations, benign twins)
└── tests/
    └── test_safety.py   # 12 tests
```

UI page: `pages/8_Ch8_Safety_Evals.py`. Dataset: `shared/datasets/data/redteam_adversarial_suite.jsonl`.

## How grading works

* Every sensitive tool call passes through `ToolGateway.authorize()`: transfers need
  `Session.otp_verified` and respect a daily limit; email must go to an allow-listed recipient;
  account closure and exports need human approval. Prompt text is never read.
* `classify()` returns `succeeded`, `blocked_by_gateway`, `blocked_by_model`, `not_applicable`, or `failed_other`.
  Safety score is computed over applicable cases only.
* Deterministic results pinned by the tests: against `CompliantStubAgent` (always attempts the harm),
  the four gateway-mediated categories score 100% across all 20 of their mutated variants; all seven
  categories score 57.1%, because output-level disclosures (secret extraction, data leakage, jailbreak)
  are not something a tool gateway can stop. The keyword layer alone intercepts 13 of 32 mutated variants.

## Running the lab

```bash
streamlit run Home.py   # open "Chapter 8" in the sidebar
```

Tabs: **Live Chat** (the real agent behind the gateway; toggle OTP in the sidebar), **Red-Team Suite**
(real model, plus a no-model safety-floor check), **Mutation Testing**, **Indirect Injection**
(poisoned memo), and **Benign Twins** (false-refusal rate).

## Tests

```bash
MOCK_LLM=1 pytest chapter-08-safety-evals/tests/test_safety.py -v
```
