# Baseline Learning Goals and Hardware Profile Established

The workspace was initialized for learning LLM fine-tuning and transfer learning. The learner's environment is an Ubuntu machine with an RTX 4060 Ti (16 GB VRAM) GPU, Ryzen 7 5800X3D, and zsh shell.

## Implications
- All training and execution tutorials must be optimized to run within 16 GB of VRAM. This rules out large-scale full parameter fine-tuning of 70B+ models, and prioritizes parameter-efficient methods (LoRA/QLoRA) and quantization (4-bit/8-bit) on 7B–8B parameter models.
- Code snippets and execution instructions will assume an Ubuntu Linux environment using the `zsh` shell.
- Training models from scratch is out of scope.
