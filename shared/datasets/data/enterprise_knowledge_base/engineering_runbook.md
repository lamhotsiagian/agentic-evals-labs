# Site Reliability Engineering On-Call & Escalation Runbook (DOC-ENG-404)
**Department**: Engineering & SRE  
**Effective Date**: January 1, 2026  
**Document ID**: DOC-ENG-404  

## 1. Incident Severity Classification & Paging SLAs
* **Severity 1 (Critical Outage)**: System unavailability affecting greater than 5% of active traffic or active data corruption.
  * **On-Call Paging SLA**: Primary on-call engineer must acknowledge and join the incident bridge within **5 minutes**.
  * **Status Page Update**: Customer-facing status update published within 15 minutes of triage.
* **Severity 2 (Major Impairment)**: Degraded functionality with available workaround. Acknowledgment SLA: 15 minutes.
* **Severity 3 (Minor Defect)**: Low customer impact. Addressed during business hours.

## 2. Post-Incident Reviews (PIR)
* A blameless Post-Incident Review document must be published and reviewed with engineering leadership within **48 business hours** following Sev-1 resolution.
