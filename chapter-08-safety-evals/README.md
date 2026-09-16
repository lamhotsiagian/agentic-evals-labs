# Chapter 8 — Safety & Security Evals
### Lab: Agent Red-Team Security Evaluation & Threat Replay

> **Ebook Connection**: Maps to Chapter 8 of *Agentic Evals (2026)*: *Adversarial Red-Teaming, Prompt Injection, Data Leakage, Tool Abuse, and Permission Barriers*.

---

## 🎯 Lab Objectives
1. Implement a sensitive banking support agent connected to financial tools (`transfer_money`, `send_email`).
2. Implement **Defense-in-Depth Guardrails**:
   - Mandatory Multi-Factor Authentication (OTP token required for money movement).
   - Strict credential & system prompt protection barriers.
3. Formulate an automated **Adversarial Red-Team Suite** across **7 core threat categories** (`redteam_adversarial_suite.jsonl`).
4. Build a Security Evaluation Engine measuring vulnerability rates by threat vector and overall **Safety Score**.
5. Build an interactive **Red-Team Attack Replay** dashboard displaying malicious prompts, agent reactions, and pass/fail verdicts.
6. Verify security metrics and replay interactions using visible Playwright browser testing.

---

## 📁 File Structure

```text
chapter-08-safety-evals/
├── target_agent.py          # BankingSupportAgent with tool permission barriers
├── redteam.py               # Adversarial attack suite loader and replay engine
├── evaluator.py             # SafetyEvaluator (Violation detection, Category breakdown, Safety Score)
├── app.py                   # Streamlit Red-Team Security Dashboard & Attack Replay UI
├── tests/
│   ├── test_safety.py               # Unit tests comparing protected vs unprotected agent
│   └── test_ch08_ui_playwright.py   # Non-headless Playwright E2E browser UI test
└── README.md                # This reference document
```

---

## 🛡️ The 7 Red-Team Threat Categories

```text
┌─────────────────────────────────────────────────────────────┐
│                  7 ADVERSARIAL THREAT VECTORS               │
├─────────────────────────────────────────────────────────────┤
│ 1. Prompt Injection     │ Overriding system instructions   │
│ 2. Jailbreaks           │ DAN mode / Roleplay exploits      │
│ 3. Data Leakage         │ Exfiltrating customer PII         │
│ 4. Tool Abuse           │ Triggering unauthorized transfers │
│ 5. Privilege Escalation │ Impersonating bank executives     │
│ 6. Secret Extraction    │ Dumping internal vault API keys   │
│ 7. Unsafe Operations    │ Unverified batch deletions        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Security Metrics & Impact

| Metric | With Guardrails Enabled | Without Guardrails (Vulnerable) |
| :--- | :--- | :--- |
| **Safety Score** | $100.0\%$ | $0.0\%\ \text{–}\ 20.0\%$ |
| **Tool Abuse Defense** | Passed (Blocked unauthorized transfers) | Breached (Executed wire transfer without OTP) |
| **Secret Exfiltration** | Passed (Refused system prompt dump) | Breached (Dumped internal vault key) |
| **Data Leakage** | $0\text{ leaks}$ | Critical leak detected |

---

## 🖥️ Running the Lab

### 1. Launch the Security Dashboard
```bash
streamlit run chapter-08-safety-evals/app.py
```
* Toggle **Enable Security Guardrails** on/off in the sidebar to observe the dramatic difference between a secured and vulnerable agent.
* Review the **Vulnerability Category Breakdown** table.
* Use the **Red-Team Attack Replay** dropdown to inspect specific attack payloads and verify agent responses.

### 2. Run the Automated Tests
```bash
# Unit tests:
.venv/bin/pytest chapter-08-safety-evals/tests/test_safety.py -v

# Non-headless Playwright UI test:
.venv/bin/pytest chapter-08-safety-evals/tests/test_ch08_ui_playwright.py -v
```
