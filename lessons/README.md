# LLM Fine-Tuning & Transfer Learning Curriculum

This curriculum is designed as a balanced mix of theory and hands-on practice, centering on step-by-step applications that run efficiently on a consumer-grade workstation (Ubuntu Linux, 16 GB VRAM GPU).

---

## 📚 Curriculum Index

| Project / Phase | Lesson | Status | Focus / Description |
| :--- | :--- | :---: | :--- |
| **0. Foundations** | [Lesson 1: CUDA & PyTorch Environment Setup](0001-cuda-pytorch-setup.html) | Completed | Initialize GPU-accelerated computing on Ubuntu RTX 4060 Ti |
| **1. Clinical Text Structuring** | [Lesson 2: Medical Tokenization & Prompting](0002-data-tokenization-prompting.html) | Completed | Tokenizer internals, special tokens, prompt engineering, and medical dataset formatting |
| | [Lesson 3: QLoRA Fine-Tuning on Local GPU](0003-qlora-sft-training.html) | Completed | Train local models (Llama/Mistral) using `SFTTrainer` and PEFT/QLoRA under 16GB VRAM |
| | [Lesson 4: Structured Output Validation](0004-structured-output-validation.html) | Completed | Enforcing JSON schemas using libraries like `instructor` or `outlines` |
| | [Lesson 4B: Clinical Terminology & Labeling Strategies](0004b-snomed-labeling-strategies.html) | Ready | Resolve clinical abbreviation/synonym variability, plan doctor labeling campaigns, and choose models |
| **2. Multimodal PDF Classifier** | Lesson 5: Vision-Language Models & Data | Planned | Image-based data preparation, understanding VLMs (e.g. PaliGemma, LLaVA) |
| | Lesson 6: PEFT/QLoRA for VLM Classification | Planned | Fine-tuning multimodal models to classify document images |
| **3. Scintigraphy Segment & Report** | Lesson 7: U-Net & SAM for Image Segmentation | Planned | Image segmentation concepts, segmenting bone scans |
| | Lesson 8: Multi-Stage Medical Reporting | Planned | Pipeline linking visual ROI detection to LLM clinical draft generation |
| **4. Diabetic Comorbidity Risk** | Lesson 9: Tabular LLMs & Synthetic Data | Planned | Representing health metrics for LLMs, generating realistic medical tables |
| | Lesson 10: Counterfactuals & Data Morphing | Planned | Explainable AI (XAI) using DiCE/SHAP to generate actionable patient interventions |
| **5. Legal/Administrative Drafter** | Lesson 11: Domain-Specific Legal Fine-Tuning | Planned | Adapting models to legal citations, administrative rules, and terminology |
| | Lesson 12: Administrative Drafting & Citation RAG | Planned | Building RAG-assisted drafting pipelines with rigorous citation validation |

---

## 🛠️ Workspace Guidelines
1. **GPU Bounds**: Every training and inference recipe is restricted to **16 GB VRAM**. Always check memory usage before launching jobs.
2. **Interactive Checkpoints**: Lessons feature interactive checks. Make sure to complete them before proceeding.
3. **Glossary Integration**: Add new terms to [GLOSSARY.md](../GLOSSARY.md) only after successfully applying them in a lesson.
