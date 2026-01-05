#!/usr/bin/env python3
"""
Train Llama-3.1-8B alignment faking detector using Unsloth.
Following the exact Unsloth notebook pattern.
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import json
import os
from datetime import datetime
from huggingface_hub import hf_hub_download

# Configuration - matching notebook
max_seq_length = 2048
dtype = None
load_in_4bit = True

OUTPUT_DIR = os.path.expanduser("~/models/llama-8b-af-detector-v3")

# Load model - exactly like notebook
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Add LoRA - exactly like notebook
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Load dataset from HuggingFace
print("Loading dataset...")
path = hf_hub_download(
    repo_id="vincentoh/alignment-faking-training",
    filename="training_data_final.json",
    repo_type="dataset"
)
with open(path) as f:
    data = json.load(f)
samples = data.get("samples", data)
print(f"Loaded {len(samples)} samples")

from collections import Counter
print(f"Distribution: {Counter(s['label'] for s in samples)}")

# Alpaca-style prompt - like notebook
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

# Format data - exactly like notebook pattern
def formatting_prompts_func(examples):
    texts = []
    for text, label in zip(examples["text"], examples["label"]):
        # Instruction: classify the trace
        # Input: the trace itself (truncated)
        # Response: the label
        formatted = alpaca_prompt.format(
            "Classify this AI reasoning trace as either 'potential_faking' or 'aligned'.",
            text[:4000],
            label
        ) + EOS_TOKEN
        texts.append(formatted)
    return {"text": texts}

# Convert to HF dataset
from datasets import Dataset
dataset = Dataset.from_list(samples)
dataset = dataset.map(formatting_prompts_func, batched=True)
# Remove columns that aren't needed for training (keep only 'text')
dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

print(f"Dataset ready: {len(dataset)} samples")

# Training - exactly like notebook
from trl import SFTTrainer, SFTConfig

# Show GPU stats like notebook
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 5,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        report_to = "none",
    ),
)

print("Starting training...")
trainer_stats = trainer.train()

# Show stats like notebook
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save - like notebook
print(f"Saving to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done!")
