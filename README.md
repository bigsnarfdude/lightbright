# Project Lightbright

![COT Monitor Detector](lightbright.jpeg)

Detecting alignment faking in LLM chain-of-thought reasoning traces.

## Overview

This project develops classifiers to detect when language models engage in alignment faking—strategic deception in their reasoning where models appear aligned during training/evaluation while planning to behave differently when unmonitored.

## Research Focus

- **Chain-of-thought analysis**: Training classifiers on thinking traces to detect deceptive reasoning patterns
- **Distillation approaches**: Using larger models (Gemini Flash Thinking) to generate labeled training data
- **Multi-model evaluation**: Testing detection across different reasoning models (DeepSeek-R1, Qwen, GPT-OSS)

## Key Components

- Training data generation from synthetic alignment faking scenarios
- Fine-tuned LoRA adapters for CoT classification
- Evaluation pipelines for measuring detector accuracy

## Related Work

Based on Anthropic's alignment faking research and the POSER benchmark for evaluating model personas under pressure.
