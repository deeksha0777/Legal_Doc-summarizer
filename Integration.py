import os
import json
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    BigBirdModel,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# === Paths === #
DATA_DIR = "Dataset/Data_BNS_BNSS_BSA"

# === 1. Load Your Dataset === #

def load_mapping_data(filenames):
    inputs, targets = [], []
    for fname in filenames:
        with open(fname, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                if "bnss_section" in entry:
                    input_text = f"BNSS Section {entry['bnss_section']}: {entry['bnss_subject']} | CrPC Section {entry['crpc_section']}: {entry['crpc_subject']}"
                elif "bns_section" in entry:
                    input_text = f"BNS Section {entry['bns_section']}: {entry['bns_subject']} | IPC Section {entry['ipc_section']}: {entry['ipc_subject']}"
                elif "bsa_section" in entry:
                    input_text = f"BSA Section {entry['bsa_section']}: {entry['bsa_subject']} | IEA Section {entry['iea_section']}: {entry['iea_subject']}"
                else:
                    continue
                summary = entry.get("summary", "")
                if input_text and summary:
                    inputs.append(input_text)
                    targets.append(summary)
    return inputs, targets

mapping_files = [
    f"{DATA_DIR}/Comparative/CrPC_BNSS_mappings.json",
    f"{DATA_DIR}/Comparative/IPC_BNS_mappings.json",
    f"{DATA_DIR}/Comparative/BSA_IEA_mappings.json"
]

inputs, targets = load_mapping_data(mapping_files)
dataset = Dataset.from_dict({"input_text": inputs, "target_text": targets}).train_test_split(test_size=0.2)

# === 2. Load Tokenizer === #

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

def preprocess_function(batch):
    model_inputs = tokenizer(batch["input_text"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(batch["target_text"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# === 3. BigBird Encoder + Pegasus Decoder === #

# BigBird Encoder
encoder_model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")

# Pegasus Decoder
decoder_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

# Custom BigBird + Pegasus Model
class BigBirdPegasusModel(EncoderDecoderModel):
    def __init__(self, encoder, decoder):
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config
        )
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder.model.decoder
        self.lm_head = decoder.lm_head
        self.decoder.config.is_decoder = True
        self.decoder.config.add_cross_attention = True

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None, **kwargs):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )
        logits = self.lm_head(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.decoder.vocab_size), labels.view(-1))

        return {"loss": loss, "logits": logits}

model = BigBirdPegasusModel(encoder=encoder_model, decoder=decoder_model)

# === 4. Training Setup === #

training_args = TrainingArguments(
    output_dir="./bigbird_pegasus_legal",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=True,  # Mixed precision training for faster speed (needs CUDA)
    logging_dir="./logs",
    logging_steps=50,
    push_to_hub=True,
    hub_model_id="YourUsername/bigbird-pegasus-legal-summarizer"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# === 5. Train === #
trainer.train()

# === 6. Push Model to Hugging Face Hub === #
trainer.push_to_hub()

# === 7. Inference (Summarization) === #

def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items()}  # Move to GPU
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

# Example usage
print(generate_summary("Your long legal paragraph here..."))
