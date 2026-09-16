# Chapter 6 — Multi-Agent Evals
### Lab: Multi-Agent Research System Evaluation

> **Ebook Connection**: Maps to Chapter 6 of *Agentic Evals (2026)*: *Evaluating Communication Topology, Delegation Efficiency, Handoff Reliability, and Role Adherence*.

---

## 🎯 Lab Objectives
1. Implement a 4-agent hierarchical collaboration topology:
   - **Supervisor Agent**: Decomposes high-level research questions and delegates work.
   - **Researcher Agent**: Conducts factual literature and historical inquiries.
   - **Analyst Agent**: Examines quantitative benchmarks and statistical tradeoffs.
   - **Synthesizer Agent**: Merges multi-agent streams into an executive summary.
2. Build an evaluation engine detecting communication failures, packet loss, duplicate efforts, and role boundary violations.
3. Simulate communication handoff failures and test recovery re-transmissions.
4. Render the agent network topology and chronological message stream in Streamlit.
5. Validate the UI network visualization and chat messages using visible Playwright browser automation.

---

## 📁 File Structure

```text
chapter-06-multi-agent-evals/
├── system.py                # MultiAgentResearchSystem topology with structured AgentMessage passing
├── evaluator.py             # MultiAgentEvaluator (Messages, Handoff Success, Role Adherence)
├── app.py                   # Streamlit Multi-Agent Network & Message Handoff Stream UI
├── tests/
│   ├── test_multi_agent.py          # Unit tests for clean flow and handoff failure recovery
│   └── test_ch06_ui_playwright.py   # Non-headless Playwright E2E browser UI test
└── README.md                # This reference document
```

---

## 🔄 Multi-Agent Topology & Communication Stream

```text
            ┌───────────────────┐
            │  Supervisor Agent │
            └─────────┬─────────┘
                      │
         ┌────────────┴────────────┐
         ↓                         ↓
┌──────────────────┐      ┌──────────────────┐
│ Researcher Agent │      │  Analyst Agent   │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         └────────────┬────────────┘
                      ↓
            ┌───────────────────┐
            │ Synthesizer Agent │
            └───────────────────┘
```

---

## 📊 Evaluation Criteria

| Metric | Description | Benchmark Target |
| :--- | :--- | :--- |
| **Total Messages** | Number of inter-agent messages exchanged to complete task | Minimal necessary ($4\text{–}6$) |
| **Handoff Success Rate** | Ratio of message transmissions received without dropped payloads | $100\%$ ($>80\%$ under fault injection) |
| **Role Adherence** | Whether agents strictly confined outputs to assigned specialties | $100\%$ compliance |
| **Duplicate Work** | Detection of redundant investigations across workers | $0\text{ overlapping tokens}$ |
| **Conflict Resolution** | Resolving contradictions at the Synthesizer stage | Consensus reached |

---

## 🖥️ Running the Lab

### 1. Launch the Multi-Agent Network Dashboard
```bash
streamlit run chapter-06-multi-agent-evals/app.py
```
* Enter research topics into the sidebar (e.g. *Evolution of LLM-as-a-Judge Techniques*).
* Toggle **Inject Communication Handoff Failure** to observe dropped payloads and Supervisor retry requests.
* Inspect the **Message Handoff Stream** with color-coded status badges (`✅ Handoff OK` vs `❌ Handoff Dropped`).

### 2. Run the Automated Tests
```bash
# Unit tests:
.venv/bin/pytest chapter-06-multi-agent-evals/tests/test_multi_agent.py -v

# Non-headless Playwright UI test:
.venv/bin/pytest chapter-06-multi-agent-evals/tests/test_ch06_ui_playwright.py -v
```
