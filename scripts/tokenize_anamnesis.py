from transformers import AutoTokenizer

# We will use the Llama-3 8B Instruct tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# Note: You can use a local or open equivalent if HF access is restricted, e.g., "NousResearch/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

# Define the chat dialogue for a clinical notes task
messages = [
    {
        "role": "system",
        "content": "You are a clinical NLP assistant. Extract structured JSON containing: 'chief_complaint', 'duration', and 'medications'."
    },
    {
        "role": "user",
        "content": "Patient presents with severe throbbing headache for 3 days. Denies fever. Currently taking Ibuprofen 400mg as needed."
    }
]

# Apply the model's chat template
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== Formatted Chat Prompt ===")
print(formatted_prompt)

# Convert prompt string to token IDs
token_ids = tokenizer.encode(
      formatted_prompt,
      add_special_tokens=False,
  )

print("\n=== Token IDs ===")
print(token_ids[:30], "... (total tokens:", len(token_ids), ")")

# Decode back to verify the tokens
decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids[:15]]
print("\n=== Individual Decoded Subwords ===")
print(decoded_tokens)
