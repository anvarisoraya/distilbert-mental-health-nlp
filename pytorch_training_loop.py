from torch.optim import AdamW
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


raw_datasets = load_dataset(
    "ourafla/Mental-Health_Text-Classification_Dataset", 
    data_files={
        "train": "mental_heath_unbanlanced.csv",
        "validation": "mental_health_combined_test.csv"
    }
)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

labels = ['Normal', 'Depression', 'Suicidal', 'Anxiety']

label_to_id = {'Normal': 0, 'Depression': 1, 'Suicidal': 2, 'Anxiety': 3}

def tokenize_function(example):
    result = tokenizer(example["text"], truncation=True)
    result["labels"] = [label_to_id[status] for status in example["status"]]
    return result


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["text", "status"])
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)


optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device


from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


import evaluate
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])
    f1_metric.add_batch(predictions=predictions, references=batch["labels"])

    
final_accuracy = accuracy_metric.compute()
final_f1 = f1_metric.compute(average="macro")

print(f"Accuracy: {final_accuracy['accuracy']:.4f}")
print(f"Macro F1-Score: {final_f1['f1']:.4f}")

output_dir = "./mental-health-bert"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)