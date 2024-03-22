# output of phind chat https://www.phind.com/search?cache=qkkcmqz2aehkiymrg9w1plfr

import os
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, get_scheduler, DistilBertTokenizerFast
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from accelerate import DataLoaderConfiguration


MODEL_OUTPUT_DIR = "./train-output"
TEST_SIZE = 0.1

# Load and concatenate the train and test datasets that have been pre-split by create-training-data.py
df = pd.read_csv("datasets/comments-train.csv")

# create train/test splits
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42, stratify=df['funny_binary'])

# Tokenize the input text using the DistilBertTokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# convert the dataset into the right format for training
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Before your map calls:
# def tokenize_function_batched(examples):
#     texts = examples["text"]  # Assuming 'text' is the column with texts to tokenize.
#     try:
#         # Explicitly prepare the batch for tokenization.
#         # This assumes 'texts' is a list of strings. If the structure is different, adjust accordingly.
#         tokenized_batch = tokenizer(texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
#         return tokenized_batch
#     except Exception as e:
#         print(f"Error with batch: {texts}")
#         raise e

def tokenize_function_nonbatched(example):
    text = example["text"]  # Now 'text' is expected to be a single string
    try:
        # Tokenize the individual text entry.
        tokenized_example = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        return tokenized_example
    except Exception as e:
        print(f"Error with text: {text}")
        raise e

# Adjust the .map calls to use batched=False
train_dataset = train_dataset.map(tokenize_function_nonbatched, batched=False)
test_dataset = test_dataset.map(tokenize_function_nonbatched, batched=False)


# Use the defined function in your map calls
train_dataset = train_dataset.map(tokenize_function_nonbatched, batched=True)
test_dataset = test_dataset.map(tokenize_function_nonbatched, batched=True)

train_dataset = train_dataset.rename_column("funny_binary", "labels")
test_dataset = test_dataset.rename_column("funny_binary", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# Define the model. It's a binary classifier so 2
NUM_LABELS_FOR_BINARY_CLASSIFIER = 2

# AutoModelForSequenceClassification is for class prediction tasks (vs generative tasks)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=NUM_LABELS_FOR_BINARY_CLASSIFIER)

# Define the compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# we train until we overfit then roll back to the one just before overfit
MAX_EPOCHS = 5

# Define the training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=MAX_EPOCHS,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    #load_best_model_at_end=True,
    metric_for_best_model="accuracy",

    # qlora quantisation can be done now but we'll do it after so we can compare the benefit qlora brings
    # adapter_type="qlora",
    # adapter_config={"r": 8, "lora_alpha": 32, "lora_dropout": 0.1},

    # wandb.com offers free training checking. WandB also relies on os.environ["WANDB_API_KEY"]
    report_to=["wandb"],
)

# Define the optimizer and learning rate scheduler

# Default is Adam and AdamW has recently shown to be better so we use that
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataset) * training_args.num_train_epochs
num_warmup_steps = int(num_training_steps * 0.1)  # 10% of training steps for warmup

# cosine scheduler with warmup is SOTA in many situations
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler)
)

# Perform k-fold cross-validation, doing just one initially
# k = 1
accuracies = []
best_accuracy = 0.0
best_model_path = None

# for fold in range(k):
    # print(f"Fold {fold + 1}/{k}")
    
    # Split the dataset for the current fold
    # train_dataset = train_df.shard(num_shards=k, index=fold)
    # eval_dataset = test_df.shard(num_shards=k, index=fold)
    
for epoch in range(training_args.num_train_epochs):
    # Train the model for one epoch
    trainer.train_dataset = train_dataset
    trainer.eval_dataset = test_dataset
    trainer.train()
    
    # Evaluate the model on the training set
    train_metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
    train_accuracy = train_metrics["train_accuracy"]

    # Evaluate the model on the test set
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    test_accuracy = test_metrics["test_accuracy"]
    
    print(f"Epoch {epoch+1}/{training_args.num_train_epochs}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {test_accuracy:.4f}")
    
    # Save the model if validation accuracy improves
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_path = trainer.state.best_model_checkpoint
        trainer.save_model(best_model_path)
    
    # Check for overfitting
    if train_accuracy > test_accuracy:
        print("Overfitting detected!")
        break
    
    accuracies.append(best_accuracy)

# Calculate the average accuracy across all folds
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average accuracy across {k} folds: {average_accuracy:.4f}")
print(f"Best Model Path: {best_model_path}")

# Now demo the model
from transformers import AutoTokenizer

# Load the trained model
model = AutoModelForSequenceClassification.from_pretrained(best_model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Array of statements to perform inference on
statements = [
    "I told Siri to call me an ambulance So Siri said ok, from now on I'll call you An Ambulance.",
    "I didn't find the joke very funny.",
    "I told my son to take out the trash. I heard screaming in the back yard only to find my other son stuck in the rubbish bin.",
    "The plot was predictable and not very engaging.",
]

# Tokenize the statements
encoded_statements = tokenizer(statements, padding=True, truncation=True, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**encoded_statements)
    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1)

# Print the predicted labels
for statement, label in zip(statements, predicted_labels):
    print(f"Statement: {statement}")
    print(f"Predicted label: {label.item()}")
    print()
