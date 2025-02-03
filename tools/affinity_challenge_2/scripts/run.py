from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
import argparse

tqdm.pandas()

def load_model_and_tokenizer():
    """Load model and tokenizer from Hugging Face"""
    model_name = "alex-apoha/esm_finetuned"  # Change this to your Hugging Face model name

    model = AutoModelForSequenceClassification.from_pretrained("alex-apoha/esm_finetuned")
    tokenizer = AutoTokenizer.from_pretrained("alex-apoha/esm_finetuned")

    model.eval()
    return model, tokenizer


def predict_affinity(sequence, model, tokenizer):
    """Predict binding affinity for a given sequence"""
    inputs = tokenizer(sequence, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits.item()  # Extract predicted value

    return prediction

def main():

    parser = argparse.ArgumentParser(description="Run TAP scraping with standardized I/O")
    parser.add_argument("--input", required=True, help="Path to the input CSV")
    parser.add_argument("--output", required=True, help="Path to the output CSV")

    args = parser.parse_args()

    print("Running predictions...")
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Load sequences
    df = pd.read_csv(args.input)#.head(10)

    # Predict affinities
    df["predicted_affinity"] = df["sequence"].apply(lambda seq: predict_affinity(seq, model, tokenizer))

    # Save results
    df[["predicted_affinity"]].to_csv(args.output, index=False)

    print("Predictions saved to", args.output)

if __name__ == "__main__":
    main()