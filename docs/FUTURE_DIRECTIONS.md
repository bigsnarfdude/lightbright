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

---

### 1.2 The Evaluation Problem

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

### 1.3 Intelligence as Gradient, Not Category

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
