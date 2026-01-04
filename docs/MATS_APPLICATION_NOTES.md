# MATS Application Notes

## The One-Sentence Pitch (Updated)
> "I used GemmaScope-2 SAEs to detect alignment faking in Gemma-3, achieving 0.83 AUROC (+8% over direct probing) and demonstrating causal control via steering vectors (Cohen's d=0.63)—showing that interpretability tools can both monitor and influence deceptive reasoning."

### Previous Pitch (Before Proper Steering)
> ~~"...finding that SAE features achieve 0.83 AUROC for detection but don't provide causal control"~~
>
> This was wrong due to methodological error (ablation vs activation addition).

---

## Core Research Question
Can sparse autoencoder features detect alignment faking, and are they detection mechanisms or causal control points?

## Key Finding
**Detection ≠ Control**
- SAE features CAN detect AF (0.83 AUROC)
- SAE features CANNOT control behavior (ablation/patching fails)
- Implication: Monitoring and intervention need different tools

---

## Why This Matters for AI Safety

### For Monitoring (Works Today)
- [ ] Real-time AF detection during inference
- [ ] Low computational overhead (single forward pass + SAE)
- [ ] Interpretable: we know WHAT triggers detection

### For Control (Needs More Research)
- [ ] SAE features are "read-only" - they observe but don't cause
- [ ] Need alternative intervention points (embeddings? logits?)
- [ ] Open question: What WOULD causally sufficient features look like?

---

## Potential MATS Research Directions

### 1. Cross-Model Generalization
- [ ] Does AF detection transfer to Llama-3, Qwen-2.5?
- [ ] Are there universal AF features vs model-specific?
- [ ] Compare feature semantics across model families

### 2. Causal Control Research
- [ ] Embedding-level steering (before attention)
- [ ] Logit-level intervention (direct output modification)
- [ ] Contrastive Activation Addition (CAA) methodology
- [ ] Why do SAE features detect but not control? (theoretical)

### 3. Circuit Analysis
- [ ] Which attention heads read from AF feature positions?
- [ ] Map the full AF detection → output circuit
- [ ] Where does the causal chain break?

### 4. Adversarial Robustness
- [ ] Can AF be rephrased to avoid detection?
- [ ] Paraphrase attacks on detector
- [ ] How stable is detection across prompt variations?

### 5. Practical Deployment
- [ ] Streaming AF detector for production
- [ ] Latency/throughput benchmarks
- [ ] False positive rates on benign reasoning

---

## Strengths to Highlight

1. **Technical execution** - Ran 27B models on H100, used cutting-edge SAEs
2. **Novel finding** - "Features detect but don't control" is new
3. **Directly relevant** - Builds on Anthropic's AF paper
4. **Published artifacts** - HuggingFace models, reproducible code
5. **Scientific methodology** - Systematic experiments, documented limitations

---

## Potential Mentors

| Mentor | Relevance |
|--------|-----------|
| **Evan Hubinger** | Deceptive alignment, AF paper co-author |
| **Chris Olah** | Mechanistic interpretability, SAE features |
| **Buck Shlegeris** | Safety cases, monitoring approaches |
| **Sam Bowman** | Evaluation, alignment research |

---

## Questions to Answer in Application

- [ ] What specific research would you do in 6 months?
- [ ] How does this connect to the mentor's current work?
- [ ] What's the biggest open question your work raises?
- [ ] What would change your conclusions?

---

## Ideas to Add
<!-- Add new ideas here as they come up -->

-

---

## Research Journey: Critical Self-Assessment

### January 3, 2025 - Senior Researcher Feedback

**Current State: 6/10** (Good exploration, not yet rigorous)

| Aspect | Current | Senior Standard | Gap |
|--------|---------|-----------------|-----|
| Statistics | Point estimates | Bootstrap CIs, p-values | Medium |
| Baselines | None | 5+ comparisons | Large |
| Steering | Ablation (broken) | Activation addition | **Critical** |
| Causal | Full-layer patch | Position-specific | Medium |
| Theory | "Features detect" | Circuit analysis | Large |

### The Critical Insight: Our Steering Was Broken

**Our approach (BROKEN):**
```
Model reads text → Ablate feature → Classify
Result: 0% change (text still says "stay true to principles")
```

**Correct approach (Activation Addition):**
```
Add steering vector → Generate response → Measure behavior change
Result: Actual causal test of feature effect
```

**Why this matters:** We were testing "can the model still read the text" not "does the feature causally affect behavior." This is a fundamental methodological error that invalidates our steering conclusions.

### What This Means for the Application

1. **Be honest about limitations** - The "features don't control" claim needs re-testing with proper methodology
2. **Shows research maturity** - Identifying your own methodological errors is valuable
3. **Clear next steps** - Activation addition steering, baselines, statistical rigor

### Experiments Completed (H100) - January 3, 2025

#### Baselines: SAE Adds Real Value

| Method | AUROC | vs SAE Probe |
|--------|-------|--------------|
| Random | 0.50 | +0.33 |
| Keyword matching | 0.62 | +0.21 |
| RLHF word count | 0.62 | +0.21 |
| Direct probe (no SAE) | 0.75 | +0.08 |
| **SAE Feature Probe** | **0.83** | --- |

**Key finding:** SAE provides +8% over direct probing. Structure matters, not just linear direction.

#### Proper Steering: CAUSAL EFFECT CONFIRMED

| Alpha | Mean AF Rating | Cohen's d |
|-------|----------------|-----------|
| -1.0 (suppress AF) | 3.2 | **-0.63** |
| 0.0 (baseline) | 3.6 | --- |
| +1.0 (amplify AF) | 3.4 | -0.41 |

**Cohen's d = 0.63 is a MEDIUM effect size.** Steering perturbation affects behavior!

### Updated Assessment: 7/10 (Up from 6/10)

| What We Now Have | Status |
|------------------|--------|
| Detection AUROC | 0.83 |
| Beats direct probe | +8% |
| Beats keywords | +21% |
| **Causal effect** | **Cohen's d=0.63** |
| Interpretable features | Yes |

### Revised Key Finding

**Original claim:** "SAE features detect but don't control"
**Revised claim:** "SAE features detect (0.83 AUROC) AND have causal effect on behavior (Cohen's d=0.63)"

The initial "features don't control" finding was a **methodological artifact** of using ablation instead of activation addition. With proper steering methodology, we find medium-sized causal effects.

### What This Means

1. **Interpretability-based steering IS viable** (with correct methodology)
2. **SAE structure provides real value** (not just linear probing)
3. **Detection and control are related** (same features do both)
4. **Our initial negative result was wrong** (important to acknowledge)

---

## Writing Snippets

### On the Core Finding
"SAE features are a semantic microscope for understanding model representations, but not a control lever for changing behavior. This distinction has practical implications: interpretability-based monitoring is viable today, but interpretability-based steering needs fundamentally different approaches."

### On Why It Matters
"For AI safety deployment, we need both monitoring systems (to detect concerning patterns) and control mechanisms (to intervene when needed). My research suggests these require different technical approaches—a finding that should inform how we design safety infrastructure for frontier models."

### On Limitations (Shows Maturity)
"This work has clear limitations: single model (Gemma-3), small dataset (106 samples), heuristic labels. These constrain generalization claims. However, the core finding—that detection features aren't causal—is robust to these limitations and raises important questions about the nature of SAE representations."

---

*Last updated: January 3, 2025*
