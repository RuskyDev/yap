from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
import json

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

def preprocess_function(examples):
    encoding = tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
    encoding['labels'] = encoding['input_ids']
    return encoding

with open('dataset.json', 'r') as f:
    data = json.load(f)

train_data = data['train']
eval_data = data['eval']

train_dataset = Dataset.from_dict({"text": [entry["text"] for entry in train_data]})
eval_dataset = Dataset.from_dict({"text": [entry["text"] for entry in eval_data]})

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=1000,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.eval()

input_text = "*he looks at her with a gentle smile*"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {key: value.to(model.device) for key, value in inputs.items()}

num_return_sequences = 3
num_beams = 5
max_length = 50

with torch.no_grad():
    generated_ids = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

for i, generated_id in enumerate(generated_ids):
    generated_text = tokenizer.decode(generated_id, skip_special_tokens=True)
    print(f"Generated Response {i+1}: {generated_text}")
