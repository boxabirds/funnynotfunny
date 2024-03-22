import argparse
import json
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def find_checkpoint_folder(epoch_output):
    checkpoint_folders = [f for f in os.listdir(epoch_output) if re.match(r'checkpoint-\d+', f)]
    if checkpoint_folders:
        return os.path.join(epoch_output, checkpoint_folders[0])
    else:
        raise ValueError(f"No checkpoint folder found in {epoch_output}")

def load_best_model(checkpoint_folder):
    with open(os.path.join(checkpoint_folder, "trainer_state.json"), "r") as f:
        trainer_state = json.load(f)
    best_model_path = trainer_state["best_model_checkpoint"]
    return AutoModelForSequenceClassification.from_pretrained(best_model_path)

def perform_inference(model, tokenizer, statements):
    encoded_statements = tokenizer(statements, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded_statements)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
    return predicted_labels

def main(args):
    checkpoint_folder = find_checkpoint_folder(args.epoch_output)
    model = load_best_model(checkpoint_folder)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    if args.comments:
        with open(args.comments, "r") as f:
            statements = f.read().splitlines()
    else:
        statements = [
            "I told Siri to call me an ambulance So Siri said ok, from now on I'll call you An Ambulance.",
            "I didn't find the joke very funny.",
            "I told my son to take out the trash. I heard screaming in the back yard only to find my other son stuck in the rubbish bin.",
            "The plot was predictable and not very engaging.",
        ]

    predicted_labels = perform_inference(model, tokenizer, statements)

    for statement, label in zip(statements, predicted_labels):
        print(f"Statement: {statement}")
        print(f"Predicted label: {label.item()}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on comments using a trained model")
    parser.add_argument("--epoch-output", default="train-output", help="Directory containing checkpoint folders")
    parser.add_argument("--comments", help="File containing comments to classify (one per line)")
    args = parser.parse_args()
    main(args)
