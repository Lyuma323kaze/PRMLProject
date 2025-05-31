from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
import torch
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt 
import os
import gc

dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")

train_dataset = dataset["train"]
split_index = len(train_dataset) // 2
sampled_train_dataset = train_dataset.select(range(split_index))

raw_dataset = sampled_train_dataset.train_test_split(
    test_size=0.1,
    seed=42
)
dataset = DatasetDict({
    "train": raw_dataset["train"],
    "validation": raw_dataset["test"]
})

def format_example(example):
    text = f"Question: {example['question']}\nContext: {example['context']['contexts'][0]}\nAnswer: {example['long_answer']}<|endoftext|>"
    return {"text": text}

dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

model_name = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./qwen2-pubmedqa",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=500,
    learning_rate=1e-5,
    optim="adamw_torch_fused",
    fp16=True,
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=True,
    metric_for_best_model="eval_loss"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

train_result = trainer.train()
gc.collect()
torch.cuda.empty_cache()

model.save_pretrained("./qwen2-pubmedqa-final")
tokenizer.save_pretrained("./qwen2-pubmedqa-final")