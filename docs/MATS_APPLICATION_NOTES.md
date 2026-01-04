# MATS Application Notes

## The One-Sentence Pitch
> "I used GemmaScope-2 SAEs to investigate alignment faking detection in Gemma-3, finding that SAE features achieve 0.83 AUROC for detection but don't provide causal control—suggesting that interpretability-based monitoring and steering require fundamentally different approaches."

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

### Experiments In Progress (H100)
- [ ] `compute_baselines.py` - Keyword, length, direct probe comparisons
- [ ] `proper_steering_vectors.py` - Activation addition (correct methodology)

### Key Questions This Raises
1. If proper steering DOES work, what does that mean for interpretability-based control?
2. If proper steering DOESN'T work, is it because features are truly non-causal, or because we need different intervention points?
3. What would a "causally sufficient" feature look like?

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
