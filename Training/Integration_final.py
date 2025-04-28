import os
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    BigBirdModel,
    BartTokenizer,
    BartModel,
    EncoderDecoderModel,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# === Paths === #
DATA_DIR = "MILDSum_Samples"  # Update with your actual path

# === 1. Load Dataset === #
def load_mildsum_data(data_dir):
    inputs, targets = [], []
    for case_dir in os.listdir(data_dir):
        case_path = os.path.join(data_dir, case_dir)
        if os.path.isdir(case_path):
            judgment_path = os.path.join(case_path, 'EN_Judgment.txt')
            summary_path = os.path.join(case_path, 'EN_Summary.txt')
            if os.path.exists(judgment_path) and os.path.exists(summary_path):
                with open(judgment_path, 'r', encoding='utf-8') as f:
                    inputs.append(f.read())
                with open(summary_path, 'r', encoding='utf-8') as f:
                    targets.append(f.read())
    return inputs, targets

inputs, targets = load_mildsum_data(DATA_DIR)
dataset = Dataset.from_dict({"input_text": inputs, "target_text": targets}).train_test_split(test_size=0.2)

# === 2. Tokenizer === #
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")  # Base to reduce memory load

def preprocess_function(batch):
    model_inputs = tokenizer(batch["input_text"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(batch["target_text"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# === 3. EncoderDecoder Model === #
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/bigbird-roberta-base",  # Encoder
    "facebook/bart-base"            # Decoder
)

# Configure model
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# === 4. Training Setup === #
training_args = TrainingArguments(
    output_dir="./bigbird_bart_legal",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50,
    push_to_hub=True,
    hub_model_id="Deeksha0777/bigbird-bart-legal-summarizer"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# === 5. Train the Model === #
trainer.train()

# === 6. Push to Hugging Face Hub === #
trainer.push_to_hub()

# === 7. Inference === #
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    model.cuda()
    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === 8. Example Inference === #
example_text = dataset["test"][0]["input_text"]
print("Generated Summary:\n", generate_summary(example_text))
