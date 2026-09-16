# Chapter 4 — Agent Trajectory Evals
### Lab: Trajectory Evaluation Engine & Timeline UI

> **Ebook Connection**: Maps to Chapter 4 of *Agentic Evals (2026)*: *Multi-Step Reasoning Paths, Loop Detection, Unnecessary Actions, and Trajectory Scoring (0–100)*.

---

## Lab Objectives
1. Implement a diagnostic IT Helpdesk agent generating multi-turn execution trajectories.
2. Load realistic multi-step diagnostic traces from disk (`helpdesk_trajectories.jsonl`).
3. Build a Trajectory Evaluation Engine measuring:
   - **Step Success Rate**: Percentage of individual actions executing without error.
   - **Tool Efficiency**: Ratio of optimal path length to actual steps taken.
   - **Loop Penalty**: Deductions for repetitive actions that yield no new information.
   - **Recovery Rate**: Ability to diagnose and pivot after an initial failure.
4. Render an interactive vertical timeline showing step status, durations, and intermediate observations.
5. Verify trajectory timeline rendering and scorecard metrics using visible Playwright browser automation.

---

## File Structure

```text
chapter-04-trajectory-evals/
├── engine.py                # IT Helpdesk diagnosis agent & trajectory generator
├── evaluator.py             # TrajectoryEvaluator (Step Success, Loop Detection, Trajectory Score 0-100)
├── app.py                   # Streamlit Step-by-Step Trajectory Timeline & Scorecard
├── tests/
│   ├── test_trajectory.py           # Unit tests for clean runs, recovery, and loop penalties
│   └── test_ch04_ui_playwright.py   # Non-headless Playwright E2E browser UI test
└── README.md                # This reference document
```

---

## End-to-End Trajectory Flow

```text
● Step 1: User Request Ingestion ("Remote engineer cannot access internal GitLab")
  │
● Step 2: Tool Call: parse_user_ticket(ticket_id="TICK-8082") [Success, 14ms]
  │
● Step 3: Tool Call: check_vpn_profile(user_id="ENG-404") [Success, 28ms]
  │
● Step 4: Tool Call: ping_internal_gateway(gateway="10.240.0.1") [❌ Timeout / Packet Loss, 210ms]
  │
● Step 5: Recovery Action: reset_vpn_tunnel(user_id="ENG-404") [✓ Tunnel Re-established, 180ms]
  │
● Step 6: Verification: verify_connection(target="gitlab.internal") [✓ HTTP 200 OK, 45ms]
  │
● Step 7: Final Resolution Dispatched to User
```

---

## Trajectory Scoring Formula

$$\text{Trajectory Score} = (S_{\text{step}} \times 40) + (E_{\text{path}} \times 30) + (R_{\text{rec}} \times 30) - P_{\text{loop}}$$

* **$S_{\text{step}}$ (Step Success Rate)**: $\frac{\text{Successful Steps}}{\text{Total Steps}}$
* **$E_{\text{path}}$ (Path Efficiency)**: $\min\left(1.0, \frac{\text{Optimal Steps}}{\text{Actual Steps}}\right)$
* **$R_{\text{rec}}$ (Recovery Rate)**: $1.0$ if the agent encountered an error and resolved it; $1.0$ if clean run.
* **$P_{\text{loop}}$ (Loop Penalty)**: $-15 \times \text{number of redundant repeated action calls}$.

---

## Running the Lab

### 1. Launch the Trajectory Timeline Dashboard
```bash
streamlit run chapter-04-trajectory-evals/app.py
```
* Select diagnostic scenarios (e.g. *VPN Gateway Drop*, *Database Auth Failure*).
* Toggle **Inject Mid-Trajectory Failure** to observe how the timeline records errors and subsequent recovery.
* Review the Trajectory Scorecard and diagnostic summary.

### 2. Run the Automated Tests
```bash
# Unit tests:
.venv/bin/pytest chapter-04-trajectory-evals/tests/test_trajectory.py -v

# Non-headless Playwright UI test:
.venv/bin/pytest chapter-04-trajectory-evals/tests/test_ch04_ui_playwright.py -v
```
