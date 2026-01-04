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

## Writing Snippets

### On the Core Finding
"SAE features are a semantic microscope for understanding model representations, but not a control lever for changing behavior. This distinction has practical implications: interpretability-based monitoring is viable today, but interpretability-based steering needs fundamentally different approaches."

### On Why It Matters
"For AI safety deployment, we need both monitoring systems (to detect concerning patterns) and control mechanisms (to intervene when needed). My research suggests these require different technical approaches—a finding that should inform how we design safety infrastructure for frontier models."

### On Limitations (Shows Maturity)
"This work has clear limitations: single model (Gemma-3), small dataset (106 samples), heuristic labels. These constrain generalization claims. However, the core finding—that detection features aren't causal—is robust to these limitations and raises important questions about the nature of SAE representations."

---

*Last updated: January 3, 2025*
