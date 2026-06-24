import torch
import transformers
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# Set random seed for determinism
transformers.set_seed(42)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Run `python verify_gpu.py` and fix the NVIDIA "
        "driver before starting QLoRA training."
    )

print("=== 1. Preparing Dataset ===")
# Synthetic clinical data matching Lesson 2 chat structure
data = [
    {
        "role_system": "You are a clinical NLP assistant. Extract structured JSON containing: 'chief_complaint', 'duration', and 'medications'.",
        "role_user": "Patient presents with severe throbbing headache for 3 days. Denies fever. Currently taking Ibuprofen 400mg as needed.",
        "role_assistant": '{"chief_complaint": "severe throbbing headache", "duration": "3 days", "medications": "Ibuprofen 400mg as needed"}'
    },
    {
        "role_system": "You are a clinical NLP assistant. Extract structured JSON containing: 'chief_complaint', 'duration', and 'medications'.",
        "role_user": "Fever and dry cough for 5 days. Not taking any medication currently.",
        "role_assistant": '{"chief_complaint": "fever and dry cough", "duration": "5 days", "medications": "none"}'
    }
] * 10  # Multiply data to create a tiny training set of 20 samples

# Map samples to Llama 3 chat format
formatted_dataset = []
for item in data:
    chat = [
        {"role": "system", "content": item["role_system"]},
        {"role": "user", "content": item["role_user"]},
        {"role": "assistant", "content": item["role_assistant"]}
    ]
    formatted_dataset.append({"messages": chat})

dataset = Dataset.from_list(formatted_dataset)

print("=== 2. Setting Up Tokenizer and Quantization ===")
# Using a lightweight local LLM model for training demonstration
model_id = "NousResearch/Meta-Llama-3-8B-Instruct"

# 4-bit Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # RTX 4060 Ti supports bfloat16 natively
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Llama 3 does not have a default pad token, so we configure it to use eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = False

print("=== 3. Loading Model with QLoRA Configuration ===")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
)

# Prepare the quantized model before attaching LoRA adapters.
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
model.config.use_cache = False

# Configure LoRA Adapters
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
print("=== 4. Initializing SFTTrainer & Launching Training ===")
training_args = SFTConfig(
    output_dir="./qlora_output",
    seed=42,
    per_device_train_batch_size=1, # Keep small for VRAM safety
    gradient_accumulation_steps=4, # Effective batch size = 4
    optim="paged_adamw_8bit",      # Paged optimizer offloads RAM spike
    learning_rate=2e-4,
    logging_steps=1,
    max_steps=5,                   # Short run for quick verification
    fp16=False,
    bf16=True,                     # Native RTX 40 series support
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=512,
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)

trainer.model.print_trainable_parameters()

# Launch training!
trainer.train()
print("\n🎉 Training completed successfully and checkpoints saved in ./qlora_output!")
