import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    TrainingArguments, 
    Trainer
)


# Load dataset
dataset = load_dataset(
    "ourafla/Mental-Health_Text-Classification_Dataset", 
    data_files={
        "train": "mental_heath_unbanlanced.csv",
        "test": "mental_health_combined_test.csv"
    }
)

labels = ['Normal', 'Depression', 'Suicidal', 'Anxiety']
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

def map_labels(example):
    example["label"] = label2id[example["status"]]
    return example

print("Mapping string labels to integers...")
dataset = dataset.map(map_labels)




checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    # Max length set to 128 to keep memory usage low and training fast
    return tokenizer(example["text"], truncation=True, max_length=128)

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # We use 'macro' average for F1 because this is a multi-class problem (4 classes)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    
    return {"accuracy": acc, "f1": f1}


model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, 
    num_labels=len(labels),
    label2id=label2id,   # Tells the model how to map integers BACK to strings
    id2label=id2label    # Useful when doing inference later!
)



training_args = TrainingArguments(
    output_dir="./mental-health-classifier",
    eval_strategy="epoch",            
    save_strategy="epoch",            
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,               
    weight_decay=0.01,                
    fp16=True,                        
    load_best_model_at_end=True       
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

final_results = trainer.evaluate()
print("Final Evaluation Results:", final_results)

trainer.save_model("./best-mental-health-model")