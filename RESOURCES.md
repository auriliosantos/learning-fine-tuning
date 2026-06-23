# Fine-Tuning & Transfer Learning Resources

## Knowledge

### Core Frameworks & Fine-Tuning Libraries
- [Hugging Face TRL (Transformer Reinforcement Learning) Documentation](https://huggingface.co/docs/trl/index)
  The standard library for instruction tuning, SFT (Supervised Fine-Tuning), DPO, etc. Use for: SFTTrainer, reward modeling.
- [Hugging Face PEFT (Parameter-Efficient Fine-Tuning) Documentation](https://huggingface.co/docs/peft/index)
  Official guide for LoRA, QLoRA, prefix tuning, and prompt tuning. Use for: configuring LoRA adapters and quantization configurations.
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
  A highly optimized framework for faster LLM fine-tuning that reduces VRAM usage significantly. Perfect for consumer GPUs (like RTX 4060 Ti).
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
  Foundational deep learning library. Use for: custom training loops, datasets, and GPU memory management.

### Articles & Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  The foundational paper on parameter-efficient fine-tuning via low-rank decomposition. Use for: understanding LoRA hyperparameters (r, alpha, dropout).
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
  Introduces 4-bit NormalFloat (NF4) quantization and double quantization. Use for: fitting large models on 16GB VRAM.

### Public Medical Datasets
- [Hugging Face Medical Datasets (e.g., `BIOMEDNLP/SciQ`, `medmcqa`, `Clinical-Question-Answering`)](https://huggingface.co/datasets?search=medical)
  A collection of open medical text datasets. Use for: transfer learning and text structuring practice.
- [MIMIC-IV-NoteDeid (Clinical Notes)](https://physionet.org/content/mimic-iv-note-deid/1.0/) or public clinical datasets on Hugging Face (e.g., [CliniNote](https://huggingface.co/datasets))
  Provides de-identified clinical notes. Use for: practicing anamnesis parsing.

## Wisdom (Communities)

- [Local/Online: r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
  Excellent community for running and training open-source models on consumer-grade GPUs. Use for: hardware discussion, quantization questions, model recommendations.
- [Hugging Face Forums](https://discuss.huggingface.co/)
  Official discussion forums. Use for: troubleshooting library-specific bugs (transformers, peft, trl).
