import torch
import transformers
from pathlib import Path
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import outlines

# Set random seed for determinism
transformers.set_seed(42)

# 1. Define the Pydantic schema matching your clinical database requirements
class ClinicalAnamnesis(BaseModel):
    chief_complaint: str = Field(description="The primary clinical complaint or symptom")
    duration: str = Field(description="The duration of the complaint")
    medications: str = Field(description="List of current medications or 'none'")

# 2. Configs
model_id = "NousResearch/Meta-Llama-3-8B-Instruct"
adapter_root = Path("./qlora_output")


def find_adapter_path(root: Path) -> Path:
    """Return the root adapter or the latest valid Trainer checkpoint."""
    if (root / "adapter_config.json").is_file():
        return root

    checkpoints = [
        path
        for path in root.glob("checkpoint-*")
        if (path / "adapter_config.json").is_file()
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No adapter_config.json found in {root} or its checkpoint-* directories."
        )

    return max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[1]))


adapter_id = find_adapter_path(adapter_root)
print(f"Using adapter: {adapter_id}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("=== 1. Loading Base Model ===")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.clean_up_tokenization_spaces = False

print("=== 2. Attaching Fine-Tuned QLoRA Adapter ===")
model = PeftModel.from_pretrained(base_model, str(adapter_id))
model.eval() # Set model to evaluation mode

print("=== 3. Wrapping Model with Outlines ===")
# Wrap the PyTorch model and tokenizer with the Outlines 1.x API.
outlines_model = outlines.from_transformers(model, tokenizer)

# 4. Define raw clinical anamnesis query
new_clinical_note = "Patient complains of sharp lower back pain since yesterday morning. Took Acetaminophen 500mg once."

messages = [
    {
        "role": "system",
        "content": (
            "You are a clinical NLP assistant. Extract structured JSON containing: "
            "'chief_complaint', 'duration', and 'medications'."
        ),
    },
    {"role": "user", "content": new_clinical_note},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("=== 4. Executing Guided Generation ===")
# Outlines constrains generation to the Pydantic schema and returns a validated model.
generated_json = outlines_model(
    prompt,
    ClinicalAnamnesis,
    max_new_tokens=128,
)
structured_data = ClinicalAnamnesis.model_validate_json(generated_json)

print("\n🎉 Output Successfully Generated and Validated!")
print("===========================================")
print("Type:", type(structured_data))
print("JSON Object:", structured_data.model_dump_json())
print("Chief Complaint:", structured_data.chief_complaint)
print("Duration:", structured_data.duration)
print("Medications:", structured_data.medications)
