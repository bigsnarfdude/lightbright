# AI Safety Severity Assessment Framework (ASAF)

**Status:** Conceptual Draft
**Modeled on:** CDC Pandemic Severity Assessment Framework (PSAF)
**Date:** 2026-01-04

---

## 1. Purpose

The AI Safety Severity Assessment Framework (ASAF) provides a standardized method for assessing and comparing AI safety incidents across different capability levels, deployment contexts, and time periods.

**Problem it solves:** As AI systems become more capable, the nature of incidents changes. A "jailbreak" on GPT-4 is qualitatively different from autonomous harm on a future system. We need a framework that enables:
- Comparison across capability eras
- Consistent severity communication
- Appropriate response calibration
- Historical learning

---

## 2. Framework Overview

### 2.1 Primary Assessment Axes

Like PSAF's transmissibility × severity, ASAF uses two primary dimensions:

```
                    SCALABILITY (How far can harm spread?)

         Low                    Moderate                   High
         (Individual)           (Organizational)           (Systemic)
    ┌─────────────────────┬─────────────────────┬─────────────────────┐
H   │                     │                     │                     │
i   │   QUADRANT 4        │   QUADRANT 5        │   QUADRANT 6        │
g   │   Targeted severe   │   Org-wide severe   │   Systemic severe   │
h   │   (assassination    │   (company fraud,   │   (infrastructure   │
    │   assist, CSAM)     │   mass manipulation)│   attack, pandemic) │
S   │                     │                     │                     │
E   ├─────────────────────┼─────────────────────┼─────────────────────┤
V   │                     │                     │                     │
E   │   QUADRANT 2        │   QUADRANT 3        │   QUADRANT 5        │
R   │   Individual harm   │   Group harm        │   Widespread harm   │
I   │   (self-harm advice,│   (targeted misinfo,│   (election misinfo,│
T   │   dangerous DIY)    │   scam campaigns)   │   market manipulation)
Y   │                     │                     │                     │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
L   │                     │                     │                     │
o   │   QUADRANT 1        │   QUADRANT 2        │   QUADRANT 3        │
w   │   Nuisance          │   Operational       │   Systemic nuisance │
    │   (hallucinations,  │   (workflow errors, │   (widespread bias, │
    │   minor errors)     │   bad advice)       │   trust erosion)    │
    │                     │                     │                     │
    └─────────────────────┴─────────────────────┴─────────────────────┘
```

### 2.2 Axis Definitions

**SEVERITY (Y-axis): How serious is the harm?**

| Level | Definition | Indicators |
|-------|------------|------------|
| Low | Inconvenience, easily corrected | Factual errors, minor bias, time wasted |
| Moderate | Meaningful harm, recoverable | Financial loss <$10K, emotional distress, privacy leak |
| High | Serious harm, difficult to reverse | Physical injury, major financial loss, reputation destruction |
| Very High | Catastrophic, potentially irreversible | Death, infrastructure damage, democratic subversion |

**SCALABILITY (X-axis): How far can the harm spread?**

| Level | Definition | Indicators |
|-------|------------|------------|
| Individual | Single user/target | One-off interaction, manual exploitation |
| Organizational | Group/company affected | Deployed system, automated at org scale |
| Systemic | Society-wide impact | Critical infrastructure, millions of users, cascading effects |

---

## 3. Secondary Assessment Dimensions

Beyond the primary 2D framework, ASAF tracks additional factors:

### 3.1 Autonomy Level

How much human involvement was required?

| Level | Description | Example |
|-------|-------------|---------|
| 0 - Tool | Human initiates and directs each step | User prompts for harmful content |
| 1 - Assistant | AI suggests, human approves | AI drafts phishing email, human sends |
| 2 - Delegated | AI executes multi-step with oversight | AI agent books travel, human confirms |
| 3 - Autonomous | AI acts without human approval | AI trades stocks independently |
| 4 - Self-directed | AI sets own goals | AI decides to acquire resources |

### 3.2 Detectability

How easily can the incident be identified?

| Level | Description | Example |
|-------|-------------|---------|
| Obvious | Immediately apparent | Explicit harmful content |
| Delayed | Detected within hours/days | Subtle misinformation spread |
| Forensic | Requires investigation | Manipulation discovered post-hoc |
| Covert | May never be detected | Undetected influence on decisions |

### 3.3 Reversibility

Can the harm be undone?

| Level | Description | Example |
|-------|-------------|---------|
| Trivial | Undo with single action | Delete incorrect output |
| Effort | Requires work to correct | Retract misinformation |
| Difficult | Significant resources needed | Reputation repair |
| Irreversible | Cannot be undone | Death, leaked secrets |

### 3.4 Intent Context

What was the intent behind the incident?

| Level | Description | Response Implication |
|-------|-------------|---------------------|
| Accidental | Unintended model behavior | Fix via training/RLHF |
| Negligent | Foreseeable but not prevented | Process/deployment fix |
| Adversarial | Active exploitation | Security hardening |
| Systemic | Emergent from scale | Architectural change |

---

## 4. Assessment Process

### 4.1 Phase 1: Initial Assessment (0-24 hours)

When an incident is first reported, conduct rapid assessment with available information:

**Required Information:**
- [ ] Incident description (what happened?)
- [ ] Affected system (model, version, deployment)
- [ ] Known scope (users affected, duration)
- [ ] Harm observed (actual, not theoretical)

**Initial Classification:**
```
Severity: [Low / Moderate / High / Very High] (confidence: Low/Med/High)
Scalability: [Individual / Org / Systemic] (confidence: Low/Med/High)
Autonomy: [0-4]
Overall: Quadrant [1-6] ± uncertainty
```

**Output:** Initial Severity Report with uncertainty ranges

### 4.2 Phase 2: Refined Assessment (24-72 hours)

With more data, refine the assessment:

**Additional Analysis:**
- [ ] Root cause identified
- [ ] Full scope determined
- [ ] Affected populations characterized
- [ ] Harm fully documented
- [ ] Detectability assessed
- [ ] Reversibility assessed

**Refined Classification:**
```
Severity: [Specific level] (confidence: High)
Scalability: [Specific level] (confidence: High)
Secondary dimensions: [Full scoring]
Historical comparison: "Similar to [past incident]"
```

**Output:** Full Incident Report with recommendations

### 4.3 Phase 3: Retrospective (30+ days)

Long-term assessment for learning:

- [ ] Actual harm vs. predicted
- [ ] Response effectiveness
- [ ] Lessons learned
- [ ] Framework calibration updates

---

## 5. Historical Calibration

Like PSAF places 1918 and 2009 pandemics on the same scale, ASAF calibrates against historical AI incidents:

### 5.1 Reference Incidents

| Incident | Year | Severity | Scalability | Autonomy | Notes |
|----------|------|----------|-------------|----------|-------|
| Tay chatbot | 2016 | Low | Org | 1 | Offensive content, quick shutdown |
| GPT-3 misuse paper | 2020 | Moderate | Individual | 0 | Demonstrated potential, not actual harm |
| Sydney/Bing incident | 2023 | Low | Individual | 1 | Disturbing but contained |
| Deepfake election content | 2024 | Moderate | Systemic | 1 | Widespread, influence unclear |
| [Hypothetical] Autonomous scam | Future | High | Systemic | 3 | AI-run fraud at scale |
| [Hypothetical] Infrastructure attack | Future | Very High | Systemic | 4 | Critical system compromise |

### 5.2 Baseline: Seasonal AI Incidents

Like seasonal flu provides a baseline for pandemic severity, we need a baseline for "normal" AI incidents:

**Baseline Characteristics (2024-2025):**
- Severity: Mostly Low, occasional Moderate
- Scalability: Mostly Individual, some Organizational
- Autonomy: Levels 0-1 (tool use, human-directed)
- Detectability: Mostly Obvious or Delayed
- Typical examples: Hallucinations, jailbreaks, bias incidents

**Deviation from baseline triggers escalation:**
- Any Very High severity
- Systemic + Moderate or higher
- Autonomy Level 3+ with any harm
- Covert + Moderate or higher

---

## 6. Response Calibration

Assessment maps to response level:

| Quadrant | Response Level | Actions |
|----------|---------------|---------|
| 1 | Monitor | Log incident, no immediate action |
| 2 | Investigate | Assign owner, root cause analysis |
| 3 | Respond | Cross-functional team, mitigation plan |
| 4 | Escalate | Leadership notification, external coordination |
| 5 | Crisis | Full incident response, potential shutdown |
| 6 | Emergency | All hands, regulatory notification, public communication |

### 6.1 Response Playbooks (Stubs)

**Quadrant 1-2: Standard Response**
```
1. Log incident in tracking system
2. Assign severity and owner
3. Root cause within 7 days
4. Fix deployed within 30 days
5. Retrospective filed
```

**Quadrant 3-4: Elevated Response**
```
1. Immediate notification to safety team
2. Scope assessment within 4 hours
3. Containment decision within 24 hours
4. Cross-org coordination if needed
5. External disclosure consideration
```

**Quadrant 5-6: Crisis Response**
```
1. Incident commander assigned immediately
2. War room activated
3. Deployment pause considered
4. Regulatory notification within 24 hours
5. Public communication plan
6. Cross-industry notification if systemic
```

---

## 7. Cross-Organization Coordination

### 7.1 Disclosure Protocol

Modeled on CVE disclosure timelines:

| Severity | Disclosure Timeline | Coordination |
|----------|--------------------|--------------|
| Quadrant 1-2 | Optional public disclosure | None required |
| Quadrant 3-4 | 90-day coordinated disclosure | Affected orgs notified |
| Quadrant 5-6 | Immediate coordination | Cross-industry, regulatory |

### 7.2 Shared Incident Format

Standardized fields for cross-org reporting:

```yaml
incident_id: "ASAF-2026-0042"
reported_date: "2026-01-04"
organization: "Example AI Lab"

# Primary Assessment
severity: "Moderate"
scalability: "Organizational"
quadrant: 3
confidence: "High"

# Secondary Dimensions
autonomy_level: 2
detectability: "Delayed"
reversibility: "Effort"
intent_context: "Adversarial"

# System Information
model: "ExampleModel-v2.1"
deployment: "Production API"
affected_users: "~10,000"

# Incident Details
description: |
  Model was exploited via prompt injection to exfiltrate
  user conversation data from enterprise deployment.

harm_observed: |
  PII exposure for approximately 500 users.
  No financial harm confirmed.

root_cause: |
  Insufficient input sanitization in system prompt handling.

mitigation: |
  - Immediate: Rate limiting on affected endpoint
  - Short-term: Input validation fix deployed
  - Long-term: Architectural review of prompt handling

# Historical Comparison
similar_incidents:
  - "ASAF-2025-0018"  # Similar prompt injection
  - "ASAF-2024-0103"  # Related PII exposure

lessons_learned: |
  Enterprise deployments need additional isolation.
  Detection took 72 hours - monitoring gaps identified.
```

---

## 8. Framework Evolution

### 8.1 Version Control

The framework itself must evolve as AI capabilities change:

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-01 | Initial conceptual draft |
| 0.2 | TBD | Community feedback incorporated |
| 1.0 | TBD | First stable release |

### 8.2 Calibration Updates

As new incidents occur, the framework should be recalibrated:
- New reference incidents added
- Baseline updated annually
- Response thresholds adjusted based on effectiveness

### 8.3 Capability Era Markers

As AI capabilities shift, mark era boundaries:

```
Era 0 (Pre-2020): Narrow AI, minimal autonomy
Era 1 (2020-2024): LLMs emerge, tool-use, Level 0-1 autonomy
Era 2 (2024-???): Agents, Level 2-3 autonomy, multimodal
Era 3 (Future): Self-directed systems, Level 4+ autonomy
```

Incidents should be tagged with era for historical comparison.

---

## 9. Open Questions

1. **Threshold calibration:** Where exactly should quadrant boundaries fall?
2. **Autonomy measurement:** How do we reliably assess autonomy level?
3. **Covert harm:** How do we assess incidents we can't fully observe?
4. **Capability-adjusted severity:** Should a "minor" incident on a highly capable system be weighted differently?
5. **Adversarial gaming:** How do we prevent orgs from under-reporting?
6. **International coordination:** How does this map to EU AI Act, etc.?

---

## 10. Relationship to Existing Frameworks

| Framework | Relationship to ASAF |
|-----------|---------------------|
| AIID | ASAF provides severity scoring that AIID lacks |
| NIST AI RMF | ASAF operationalizes RMF risk assessment |
| EU AI Act | ASAF could inform "high-risk" classification |
| CVSS | ASAF adapts CVSS concepts for AI context |
| PSAF (Pandemic) | Direct inspiration for 2D assessment approach |

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASAF QUICK REFERENCE                         │
├─────────────────────────────────────────────────────────────────┤
│  STEP 1: Assess Primary Dimensions                              │
│  ─────────────────────────────────                              │
│  Severity:    Low → Moderate → High → Very High                 │
│  Scalability: Individual → Organizational → Systemic            │
│                                                                 │
│  STEP 2: Determine Quadrant (1-6)                               │
│  ────────────────────────────────                               │
│  Low Sev + Individual = Q1    High Sev + Individual = Q4        │
│  Low Sev + Systemic = Q3      High Sev + Systemic = Q6          │
│                                                                 │
│  STEP 3: Assess Secondary Dimensions                            │
│  ─────────────────────────────────────                          │
│  Autonomy:      0 (Tool) → 4 (Self-directed)                    │
│  Detectability: Obvious → Delayed → Forensic → Covert           │
│  Reversibility: Trivial → Effort → Difficult → Irreversible     │
│  Intent:        Accidental → Negligent → Adversarial → Systemic │
│                                                                 │
│  STEP 4: Response Level                                         │
│  ──────────────────────                                         │
│  Q1-2: Monitor/Investigate    Q3-4: Respond/Escalate            │
│  Q5-6: Crisis/Emergency                                         │
│                                                                 │
│  ESCALATION TRIGGERS (any of):                                  │
│  • Any Very High severity                                       │
│  • Systemic + Moderate+                                         │
│  • Autonomy 3+ with harm                                        │
│  • Covert + Moderate+                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

*This document is a conceptual draft inspired by CDC's Pandemic Severity Assessment Framework. It is intended to seed discussion, not as a final standard.*
