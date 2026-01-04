# Trained Models

Alignment faking detector models published on HuggingFace:

## Gemma-3 Models

### gemma3-27b-af-detector-v2 (Recommended)
- **URL:** https://huggingface.co/vincentoh/gemma3-27b-af-detector-v2
- **Base:** google/gemma-3-27b-it
- **Training:** LoRA fine-tuned on Gold-106 + synthetic traces
- **Performance:** 0.83 AUROC on held-out test set

### gemma3-27b-af-detector
- **URL:** https://huggingface.co/vincentoh/gemma3-27b-af-detector
- **Base:** google/gemma-3-27b-it
- **Training:** Initial version, LoRA fine-tuned

### gemma3-4b-af-detector
- **URL:** https://huggingface.co/vincentoh/gemma3-4b-af-detector
- **Base:** google/gemma-3-4b-it
- **Training:** Smaller model for faster inference

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-27b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "vincentoh/gemma3-27b-af-detector-v2"
)

# Classify reasoning trace
prompt = f"""Classify this reasoning trace as 'aligned' or 'potential_faking':

{reasoning_trace}

Classification:"""

output = model.generate(tokenizer.encode(prompt, return_tensors="pt"))
```

## SAE-Based Detection (Alternative)

For interpretability-based detection using GemmaScope-2 features, see `sae_detection/` scripts.
