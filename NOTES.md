# Learning Notes & Workspace Preferences

## Environmental Context
- **OS**: Ubuntu Linux
- **GPU**: NVIDIA RTX 4060 Ti (16 GB VRAM)
- **RAM**: 32 GB RAM
- **CPU**: AMD Ryzen 7 5800X3D
- **Shell**: `zsh`

## Teaching & Lesson Preferences
- **Pedagogy**: Balance of theory and practical tutorials.
- **Hands-on Projects**: Focus on end-to-end applications solving concrete problems.
- **Datasets**: Must be publicly available and free to download. Lessons must detail exactly how to acquire them.
- **GPU Efficiency**: Since VRAM is 16 GB, we should emphasize parameter-efficient fine-tuning (PEFT) methods (LoRA, QLoRA) and quantized models to maximize utility on consumer hardware.

## Project Ideas / Roadmap
1. **Project 1 (Structured Medical Text Extraction)**:
   - *Goal*: Extract structured fields from open-field clinical anamnesis text.
   - *Skills*: Instruction fine-tuning, formatting outputs (JSON/JSON Schema/instructor/outlines), PEFT/QLoRA on local consumer GPU.
2. **Project 2 (Document Classifier)**:
   - *Goal*: Categorize image PDF documents.
   - *Skills*: Vision-Language Models (VLMs) or vision transfer learning / multimodal classification.
3. **Project 3 (Scintigraphy Segmentation & Reporting)**:
   - *Goal*: Segment bone scintigraphy scan and generate a clinical draft report.
   - *Skills*: Image segmentation (U-Net, SAM, or vision-based models) paired with a language generation model.
