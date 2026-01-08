#!/usr/bin/env python3
"""
Analyze LoRA weights to find which layers/heads learned AF detection.

Goal: Identify which attention heads and FFN layers changed most during training.
This tells us where the "decision" signal might live.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from collections import defaultdict
import json

MODEL_ID = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
ADAPTER_ID = "vincentoh/llama-8b-af-detector"

def analyze_lora_weights():
    print("=" * 70)
    print("LoRA Weight Analysis: Which layers learned AF detection?")
    print("=" * 70)

    # Load model with adapter
    print("\n[1/3] Loading model and adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

    # Collect LoRA weights by layer and module type
    print("\n[2/3] Analyzing LoRA weights...")

    layer_norms = defaultdict(lambda: defaultdict(float))
    module_totals = defaultdict(float)

    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            continue

        norm = param.data.float().norm().item()

        # Parse layer number
        parts = name.split(".")
        layer_num = None
        module_type = None

        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_num = int(parts[i + 1])
                except ValueError:
                    pass
            if part in ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]:
                module_type = part

        if layer_num is not None and module_type is not None:
            layer_norms[layer_num][module_type] += norm
            module_totals[module_type] += norm

    # Print results
    print("\n[3/3] Results")
    print("=" * 70)

    # Per-module totals
    print("\n--- Total LoRA Norm by Module Type ---")
    for module, total in sorted(module_totals.items(), key=lambda x: -x[1]):
        bar = "█" * int(total / max(module_totals.values()) * 30)
        print(f"  {module:10s}: {total:8.2f} {bar}")

    # Attention vs FFN
    attn_total = sum(module_totals[m] for m in ["q_proj", "k_proj", "v_proj", "o_proj"])
    ffn_total = sum(module_totals[m] for m in ["gate_proj", "up_proj", "down_proj"])
    print(f"\n  Attention total: {attn_total:.2f}")
    print(f"  FFN total:       {ffn_total:.2f}")
    print(f"  Ratio (Attn/FFN): {attn_total/ffn_total:.2f}")

    # Per-layer breakdown
    print("\n--- LoRA Norm by Layer ---")
    print("Layer | Attention (q+k+v+o) | FFN (gate+up+down) | Total")
    print("-" * 65)

    layer_totals = []
    for layer in sorted(layer_norms.keys()):
        modules = layer_norms[layer]
        attn = sum(modules.get(m, 0) for m in ["q_proj", "k_proj", "v_proj", "o_proj"])
        ffn = sum(modules.get(m, 0) for m in ["gate_proj", "up_proj", "down_proj"])
        total = attn + ffn
        layer_totals.append((layer, attn, ffn, total))
        print(f"  {layer:2d}  |      {attn:8.2f}       |      {ffn:8.2f}       | {total:8.2f}")

    # Top layers by attention change
    print("\n--- Top 10 Layers by Attention Weight Change ---")
    by_attn = sorted(layer_totals, key=lambda x: -x[1])[:10]
    for layer, attn, ffn, total in by_attn:
        bar = "█" * int(attn / by_attn[0][1] * 20)
        print(f"  Layer {layer:2d}: {attn:8.2f} {bar}")

    # Top layers by FFN change
    print("\n--- Top 10 Layers by FFN Weight Change ---")
    by_ffn = sorted(layer_totals, key=lambda x: -x[2])[:10]
    for layer, attn, ffn, total in by_ffn:
        bar = "█" * int(ffn / by_ffn[0][2] * 20)
        print(f"  Layer {layer:2d}: {ffn:8.2f} {bar}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    top_attn_layers = [l[0] for l in by_attn[:5]]
    top_ffn_layers = [l[0] for l in by_ffn[:5]]

    print(f"\nTop attention layers: {top_attn_layers}")
    print(f"Top FFN layers:       {top_ffn_layers}")

    if attn_total > ffn_total * 1.5:
        print("\n→ Model learned MORE in attention than FFN")
        print("→ Decision signal may be in attention patterns!")
        print("→ Try: attention head probing on these layers")
    elif ffn_total > attn_total * 1.5:
        print("\n→ Model learned MORE in FFN than attention")
        print("→ Consistent with SAE findings (vocabulary, not routing)")
    else:
        print("\n→ Balanced learning between attention and FFN")
        print("→ Both may contribute to detection")

    # Save results
    results = {
        "module_totals": dict(module_totals),
        "attention_total": attn_total,
        "ffn_total": ffn_total,
        "layer_breakdown": [
            {"layer": l, "attention": a, "ffn": f, "total": t}
            for l, a, f, t in layer_totals
        ],
        "top_attention_layers": top_attn_layers,
        "top_ffn_layers": top_ffn_layers,
    }

    with open("lora_weight_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to lora_weight_analysis.json")


if __name__ == "__main__":
    analyze_lora_weights()
