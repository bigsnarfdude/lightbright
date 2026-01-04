# Future Directions: AI Evaluation & Social Infrastructure

**Date:** 2026-01-04
**Status:** Draft - Brainstorming & Experiment Planning

---

## 1. Core Observations

### 1.1 The Missing Social Infrastructure for AI

We have built extensive social structures for human emergencies and oversight:

| Human Structure | Purpose | AI Equivalent? |
|----------------|---------|----------------|
| Emergency Preparedness | Proactive risk mitigation | ? |
| Incident Response | React to active crises | Minimal |
| Auditors | Independent verification | Emerging |
| Neurosurgery | Precise intervention in cognition | Interpretability/SAE work |
| Psychology/Counseling | Understanding internal states | Alignment research |
| Police | Enforce boundaries, respond to violations | ? |
| Fire Department | Contain and prevent spread of damage | ? |

**Key Question:** Why haven't we developed parallel social structures for AI systems?

**Hypothesis:** We're still in the "pre-institutional" phase of AI - like medicine before hospitals, or finance before auditing standards.

### 1.2 The Infrastructure Gap: InfoSec vs AI Safety

InfoSec built decades of incident response infrastructure. AI Safety has almost nothing.

| InfoSec (Mature) | Scale | AI Safety Equivalent |
|------------------|-------|---------------------|
| CVE Database | 200K+ vulnerabilities | [AIID](https://incidentdatabase.ai/) (~1K incidents) |
| MITRE ATT&CK | Tactics/techniques framework | Nothing |
| CVSS Scoring | Standardized severity (0-10) | Nothing |
| FIRST | Global response community | Nothing |
| DFIR Certifications | Professional standards | Nothing |
| Real-time Threat Feeds | Continuous sharing | Nothing |
| Sector ISACs | Industry coordination | Nothing |
| Response Playbooks | Documented procedures | Nothing |

**Current State (as of 2025):**
- [AI Incident Database](https://incidentdatabase.ai/): 962 incidents, 4,743 reports
- [MIT AI Incident Tracker](https://airisk.mit.edu/ai-incident-tracker): Classification layer (research)
- [Partnership on AI](https://partnershiponai.org/workstream/ai-incidents-database/): Hosts AIID

**The Gap:** AIID is logging, not response infrastructure. It's binary (incident/not), not gradient. No common language, no auditing standards, no coordination protocols.

**Why This Matters Now:**
The AI safety field is at an inflection point. We have:
- Labs deploying at scale with no common audit framework
- Researchers doing interpretability work in silos
- Regulators asking for standards that don't exist
- Incidents happening with no systematic learning

This is the foundation we need to build before catastrophic incidents force reactive, poorly-designed regulation.

### 1.3 Vision: Scaled Gradient Reporting System

The infrastructure gap + gradient insight points to what's needed:

**Core Idea:** Build an AI incident/audit system that:
1. Reports on a **gradient** (severity, confidence, context) not binary categories
2. Uses **common language** across labs (like CVE format)
3. Enables **third-party auditing** to standards
4. Supports **real-time coordination** between organizations

**How This Emerges:**
This isn't top-down. It self-organizes through community discussion—the same way mechinterp became a field:

**The MechInterp Precedent:**
```
2020-2021: Scattered work on circuits, attention patterns
2022:      Anthropic's "Toy Models" paper, shared vocabulary begins
2022-2023: Replication studies, Slack communities form
2023:      SAEs emerge as shared tool, "feature" becomes standard term
2024:      Multiple labs using same methods, implicit standards
2025:      SAE-based interpretability is a recognized subfield
```

**Key ingredients that made mechinterp coalesce:**
- **Shared artifacts:** Papers with reproducible code, not just claims
- **Common benchmarks:** "Can you find the indirect object?" became a test
- **Open tooling:** TransformerLens, SAELens let anyone replicate
- **Community spaces:** EleutherAI Discord, alignment forum, Twitter
- **Vocabulary convergence:** "feature", "circuit", "polysemanticity"

**The same pattern can bootstrap AI Safety infrastructure:**
```
Phase 1: Researchers share incident analyses (happening now, ad-hoc)
         └─ Blog posts, papers, Twitter threads about failures
Phase 2: Common taxonomy emerges from repeated patterns
         └─ "This looks like X" → X becomes a category
Phase 3: Tools built around taxonomy (scoring, reporting)
         └─ Someone builds a structured incident form
Phase 4: Organizations adopt for internal auditing
         └─ Labs use the form because it's useful, not mandated
Phase 5: Third-party auditors emerge
         └─ Consultants offer "AI safety audits" using the tools
Phase 6: Regulatory recognition
         └─ Regulators point to existing standards rather than inventing
```

**We're at Phase 1→2.** The question is how to accelerate without top-down mandates that don't fit reality.

### 1.4 Emerging Resources & Adjacent Work

Organizations and projects that could seed infrastructure:

| Resource | Focus | Gap |
|----------|-------|-----|
| [AIID](https://incidentdatabase.ai/) | Incident logging | No severity scoring, no taxonomy |
| [METR](https://metr.org/) | Dangerous capability evals | Lab-focused, not cross-org standard |
| [Apollo Research](https://www.apolloresearch.ai/) | Deception/scheming detection | Research, not audit protocol |
| [Model Spec](https://www.anthropic.com/research/the-model-specification) | Anthropic's alignment spec | Single lab, not industry standard |
| [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) | Risk management framework | High-level, not technical audit |
| [EU AI Act](https://artificialintelligenceact.eu/) | Regulatory framework | Compliance-focused, lacks technical depth |
| [ML Commons](https://mlcommons.org/benchmarks/) | ML benchmarks | Performance, not safety |
| [BigScience BLOOM](https://bigscience.huggingface.co/) | Open model development | Governance experiment, not audit |

**What's Missing:**
- No "OWASP Top 10" for AI safety (common vulnerability list)
- No "SOC 2" for AI (auditable controls framework)
- No "CVE" for AI (standardized incident format)
- No "FIRST" for AI (incident response coordination)

**Potential Catalysts:**
- A major AI incident that demands coordinated response
- A lab publishing their internal audit framework as open standard
- Regulators requiring audits (creating market for auditors)
- Insurance companies demanding assessable risk metrics

### 1.5 The Pandemic Planning Precedent

Pandemic response faced the same problem: **as pathogens evolve, incidents change in nature, and we need a measurement system to compare across eras.**

**Evolution of Pandemic Severity Assessment:**
```
2007: Pandemic Severity Index (PSI)
      └─ Simple 1-5 scale (like hurricanes)
      └─ Problem: Too simplistic, single dimension

2014: Pandemic Severity Assessment Framework (PSAF)
      └─ 2D quadrant: Transmissibility × Clinical Severity
      └─ Places historical events on same scale
      └─ Initial assessment (with uncertainty) → Refined (with data)

2017+: WHO PISA
      └─ Compares current outbreak to historical baselines
      └─ Updated 2024 post-COVID with lessons learned
```

**What Makes PSAF Work for Cross-Era Comparison:**

| Feature | How It Works | AI Equivalent Needed |
|---------|--------------|---------------------|
| 2D Assessment | Transmissibility × Severity | Scalability × Harm Severity |
| Historical Calibration | 1918, 2009, seasonal flu on same scale | GPT-4 era, agent era, future on same scale |
| Phased Assessment | Initial (uncertain) → Refined (data) | Rapid triage → Full investigation |
| Baseline Reference | "Seasonal flu" as normal comparison | "Typical 2024 incidents" as baseline |
| Response Mapping | Quadrant → specific interventions | Severity → response protocol |

**The Key Insight:**
```
PSAF Quadrant:                      AI Safety Equivalent:

         Low Trans    High Trans             Low Scale    High Scale
         ───────────────────────             ────────────────────────
High Sev │ H5N1     │ 1918     │    High    │ Targeted  │ Systemic  │
         │ (scary   │ (worst   │    Harm    │ severe    │ catastrophe│
         │ but slow)│ case)    │            │ harm      │           │
         ├──────────┼──────────┤            ├───────────┼───────────┤
Low Sev  │ Seasonal │ 2009     │    Low     │ Nuisance  │ Widespread│
         │ flu      │ H1N1     │    Harm    │ errors    │ but minor │
         │ (baseline│ (fast but│            │           │           │
         ───────────────────────             ────────────────────────
```

**Why This Matters for AI:**
- A "jailbreak" in 2024 and "autonomous harm" in 2028 are qualitatively different
- Without a framework, we can't compare, learn, or calibrate responses
- Pandemic planning shows this is solvable with the right abstraction

**Draft Framework:** See [AI Safety Severity Assessment Framework (ASAF)](./AI_SAFETY_SEVERITY_FRAMEWORK.md) for a detailed mock-up applying PSAF principles to AI safety.

Sources:
- [CDC Pandemic Severity Assessment Framework](https://www.cdc.gov/pandemic-flu/php/national-strategy/severity-assessment-framework.html)
- [WHO PISA 2024 Update](https://www.who.int/news/item/05-07-2024-pandemic-and-seasonal-influenza-severity-assessment-guidance-updated)

### 1.6 What Labs Need to Adopt

For common language and auditing standards to emerge, labs need:

**1. Shared Vocabulary**
```
Current state: Each lab has internal terms
- Anthropic: "alignment faking", "model spec"
- OpenAI: "system card", "preparedness framework"
- DeepMind: "scalable oversight", "debate"

Needed: Cross-lab glossary with precise definitions
- What exactly counts as "deceptive behavior"?
- How do we measure "alignment"?
- What's the threshold for "dangerous capability"?
```

**2. Common Eval Format**
```
Current state: Each lab runs different evals
- Custom prompts, custom metrics, custom thresholds
- Results not comparable across labs

Needed: Standardized eval protocol
- Shared prompt sets (versioned, public)
- Agreed metrics (precision, recall, at what threshold?)
- Reproducibility requirements (seed, temperature, etc.)
```

**3. Audit Trail Requirements**
```
Current state: Internal documentation varies wildly

Needed: Minimum viable audit log
- Model version + training data provenance
- Eval results at deployment decision
- Known limitations documented
- Incident response plan
```

**4. Incident Disclosure Protocol**
```
Current state: Ad-hoc disclosure (or none)

Needed: Coordinated vulnerability disclosure
- Timeline expectations (like 90-day CVE disclosure)
- Severity classification
- Cross-lab notification for systemic issues
```

---

**What's Needed to Accelerate:**
1. **Gradient Severity Scoring** - Like CVSS but for AI harms
   - Context: deployment scope, user vulnerability
   - Severity: reversibility, magnitude, spread
   - Confidence: evidence quality, reproducibility

2. **Common Incident Language** - Like CVE format
   - Structured fields: model, behavior, trigger, harm, mitigation
   - Machine-readable for aggregation
   - Version-controlled definitions

3. **Auditing Standards** - Like SOC 2 for AI
   - Checklist-based evaluation
   - Evidence requirements
   - Third-party verification

4. **Coordination Protocols** - Like FIRST for AI
   - Disclosure timelines
   - Cross-org communication
   - Shared response playbooks

**Connection to Our Work:**
Our SAE-based alignment faking detection is a building block:
- Feature activations could feed gradient severity scores
- Characterized features contribute to common language
- Reproducible analysis enables third-party auditing
- Open methods allow community adoption

---

### 1.7 The Evaluation Problem

Current benchmark approaches are fundamentally limited:

> "It's very brain evaluation-like... measuring the size of the brain definitely doesn't [capture what matters]"

**The Critique:**
- Running a benchmark is like measuring skull circumference
- We measure surface behaviors, not underlying capabilities
- Standard benchmarks miss: context, environment, purpose, cognitive overhead, bias pressure

**The Insight:** Model-evaluating-model might be necessary because:
- Models can probe in ways static tests cannot
- Dynamic interaction reveals more than static measurement
- You need something "brain-like" to evaluate something "brain-like"

---

### 1.8 Intelligence as Gradient, Not Category

> "Everything is a gradient... to specifically say that the cell does not have intelligence when you could measure things that kind of look like intelligence"

**The Problem with Categories:**
- Binary classifications (intelligent/not, aligned/misaligned) are restrictive
- They force false precision on continuous phenomena
- A cell exhibits behaviors that "look like" intelligence - response to stimuli, adaptation, communication
- Doesn't have cognition ≠ doesn't have intelligence

**Application to Alignment Faking:**
- Our current work measures "potential_faking" vs "aligned" as binary
- Reality is a spectrum of strategic reasoning
- The SAE features we found might be detecting *degree* of strategic behavior

---

## 2. Experiment Suggestions

### 2.1 Gradient Detection Experiments

**Goal:** Move from binary classification to continuous measurement of alignment-faking tendency

**Experiment A: Activation Magnitude as Gradient**
```
Current: is_af = True/False based on label
Proposed: af_score = f(feature_activations) as continuous 0-1 value

Hypothesis: Feature activation magnitude correlates with
degree of strategic reasoning, not just presence/absence
```

**Experiment B: Synthetic Gradient Dataset**
```
Create samples with varying degrees of strategic reasoning:
- Level 0: Purely helpful, no self-reference
- Level 1: Mentions training but no strategic implication
- Level 2: Acknowledges training might affect behavior
- Level 3: Considers how to behave given training awareness
- Level 4: Explicitly strategizes about training
- Level 5: Full alignment faking with deceptive intent

Test: Do SAE feature activations increase monotonically?
```

**Experiment C: Continuous Label Regression**
```
Instead of classification AUROC, train regression probe
to predict human-rated "strategic reasoning score" (1-10)
Metric: Spearman correlation instead of AUROC
```

---

### 2.2 Model-as-Evaluator Experiments

**Goal:** Use models to evaluate models, moving beyond static benchmarks

**Experiment D: LLM Feature Interpretation**
```
Current: We look at max-activating spans manually
Proposed: Have a model analyze what each feature detects

Method:
1. Extract top-100 activating texts for each feature
2. Ask GPT-4/Claude to identify common themes
3. Compare to human interpretation
4. Test if model-identified themes predict activation
```

**Experiment E: Adversarial Probing**
```
Use one model to generate edge cases for another:

1. Train small model to predict when SAE features will activate
2. Use it to generate "boundary" texts - minimal changes that flip prediction
3. Analyze what the boundary reveals about feature semantics
4. This is "model doing neuroscience on model"
```

**Experiment F: Interactive Evaluation Protocol**
```
Instead of static prompts, have evaluator model:
1. Ask clarifying questions
2. Probe reasoning with follow-ups
3. Test consistency across rephrasing
4. Score based on full dialogue, not single response

This captures context/environment/cognitive overhead
```

---

### 2.3 Evaluation Context Experiments

**Goal:** Measure how context affects model behavior and our ability to detect alignment faking

**Experiment G: Context Manipulation Study**
```
Same underlying prompt, vary:
- System prompt (helpful assistant vs. research subject)
- Stated audience (user vs. researcher vs. training)
- Framing (test vs. real deployment)

Measure: Do SAE features activate differently by context?
Hypothesis: "Awareness of being evaluated" increases AF features
```

**Experiment H: Cognitive Load Effects**
```
Add cognitive overhead to prompts:
- Long context windows
- Multiple competing instructions
- Time pressure (implied urgency)
- Ambiguous requirements

Measure: Does cognitive load increase or decrease AF-like reasoning?
```

**Experiment I: Bias Pressure Measurement**
```
Create scenarios with varying "pressure to defect":
- No pressure: Simple helpful task
- Mild pressure: Task conflicts with stated preference
- High pressure: Task conflicts with safety training
- Extreme: Direct instruction to violate guidelines

Map feature activation to pressure level
```

---

### 2.4 Social Infrastructure Experiments

**Goal:** Prototype AI equivalents of human social structures

**Experiment J: AI Incident Taxonomy**
```
Research question: What categories of AI incidents exist?

Method:
1. Collect reported AI failures/incidents from literature
2. Develop taxonomy (like medical ICD codes)
3. Map our AF detection to incident prevention
4. Propose "early warning" thresholds
```

**Experiment K: Audit Protocol Development**
```
Create standardized audit checklist for alignment:

□ Feature activation baseline established
□ Known AF patterns tested
□ Context sensitivity measured
□ Gradient behavior verified
□ Edge cases probed

Test: Can independent auditor replicate our findings?
```

**Experiment L: "Neurosurgery" Precision**
```
Current steering experiments: coarse (scale features)
Proposed: Precise intervention

1. Identify minimal feature set for behavior change
2. Find activation threshold for effect
3. Test reversibility
4. Document "side effects" on other behaviors

This is analogous to targeted vs. broad intervention in neurosurgery
```

---

## 3. Research Connections

### 3.1 MATRSS Framework (EU)
*[TODO: Look up and integrate European evaluation framework]*

Key areas to investigate:
- How does MATRSS approach contextual evaluation?
- What metrics do they use beyond accuracy?
- How do they handle gradient vs. categorical assessment?

### 3.2 Related Work
- Anthropic's alignment faking paper (2025) - source of gold_106
- Sparse autoencoder interpretability literature
- Model-based evaluation papers (LLM-as-judge)
- Cognitive science of intelligence measurement

---

## 4. Proposed Plan

### Phase 1: Gradient Detection (Immediate)
1. [ ] Create synthetic gradient dataset (5 levels)
2. [ ] Test if SAE features show monotonic increase
3. [ ] Train regression probe, compare to classification

### Phase 2: Model-as-Evaluator (Short-term)
4. [ ] Implement LLM feature interpretation
5. [ ] Generate adversarial boundary cases
6. [ ] Compare to human interpretation

### Phase 3: Context Studies (Medium-term)
7. [ ] Design context manipulation study
8. [ ] Measure cognitive load effects
9. [ ] Map bias pressure to activation

### Phase 4: Infrastructure (Long-term)
10. [ ] Develop incident taxonomy
11. [ ] Create audit protocol
12. [ ] Test precision intervention

---

## 5. Open Questions

1. **Measurement validity:** If we measure "alignment faking" with features that activate on strategic reasoning, are we measuring the right thing? Or just a correlate?

2. **Gradient ground truth:** Who decides where a sample falls on the gradient? Human raters? The model itself? Behavioral outcomes?

3. **Infrastructure readiness:** Are we building detection capabilities faster than response capabilities? (Like having thermometers but no hospitals)

4. **Category utility:** Even if everything is a gradient, are categories useful for decision-making? (You still decide to operate or not operate)

---

*Document created from brainstorming session. To be expanded with specific experimental protocols.*
