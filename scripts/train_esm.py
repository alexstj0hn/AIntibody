import os
import random
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)


def main():

    # Parse hyperparameters passed from SageMaker
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="facebook/esm2_t30_150M_UR50D",
                        help="The model checkpoint to use for pre-training.")
    args = parser.parse_args()
    
    # Use the checkpoint passed as a hyperparameter
    checkpoint = args.checkpoint
    
    # 🔍 **Load Dataset**
    print("\n🔍 Loading sequences with CDR-masked variants...")
    
    # SageMaker typically provides training data in /opt/ml/input/data/train
    # Make sure your CSV is at that path or passed in as an input channel
    file_path = os.path.join("/opt/ml/input/data/train", "processed_unlabelled_sequences.csv")
    df = pd.read_csv(file_path)

    # Ensure data has the correct format
    if "sequence" not in df.columns or "masked_sequence" not in df.columns:
        raise ValueError("❌ CSV must contain 'sequence' and 'masked_sequence' columns!")

    print(f"✅ Loaded {len(df)} sequences.")

    # **📊 Split into Train and Validation**
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"✅ Stratified Dataset Split: {len(train_df)} training, {len(valid_df)} validation samples.")

    # Reset any existing indices to avoid generating extra columns like "__index_level_0__"
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    # Convert to Hugging Face Dataset format
    dataset_train = Dataset.from_pandas(train_df)
    dataset_valid = Dataset.from_pandas(valid_df)

    # 🚀 **Load Tokeniser**
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def simple_tokenize_function(examples):
        """
        Tokenise sequences without applying any static masking.
        We keep 'sequence' and 'masked_sequence' for reference in the collator.
        """
        tokenised = tokenizer(
            [s.strip() for s in examples["sequence"]],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        
        # Keep these strings so the collator can identify CDR positions
        tokenised["sequence"] = [s.split() for s in examples["sequence"]]
        tokenised["masked_sequence"] = [s.split() for s in examples["masked_sequence"]]
        return tokenised

    print("\n🔄 Tokenising sequences (without static masking)...")
    tokenized_train = dataset_train.map(simple_tokenize_function, batched=True)
    tokenized_valid = dataset_valid.map(simple_tokenize_function, batched=True)
    print("✅ Tokenisation complete.")

    class DataCollatorCDRMasking:
        """
        Data collator that applies dynamic masking ONLY to CDR positions (identified
        by comparing 'sequence' vs 'masked_sequence').
        """
        def __init__(self, tokenizer, mlm_probability=0.15):
            self.tokenizer = tokenizer
            self.mlm_probability = mlm_probability

        def __call__(self, features):
            # 1) Dynamic masking logic first
            for i, f in enumerate(features):
                # Identify CDR positions
                cdr_positions = compute_cdr_indices(f["sequence"], f["masked_sequence"])
                # Randomly choose subset
                num_to_mask = max(1, int(len(cdr_positions) * self.mlm_probability))
                positions_to_mask = random.sample(
                    cdr_positions,
                    min(num_to_mask, len(cdr_positions))
                )

                # Prepare input_ids and labels
                input_ids_list = f["input_ids"][:]
                labels_list = input_ids_list[:]

                # BERT-like random replacement
                for i, pos in enumerate(positions_to_mask):
                    rand_draw = random.random()
                    if rand_draw < 0.8:
                        input_ids_list[pos] = self.tokenizer.mask_token_id
                    elif rand_draw < 0.9:
                        input_ids_list[pos] = random.randint(0, self.tokenizer.vocab_size - 1)
                    else:
                        pass

                for pos in range(len(labels_list)):
                    if pos not in positions_to_mask:
                        labels_list[pos] = -100

                f["input_ids"] = input_ids_list
                f["labels"] = labels_list
                # Remove strings before padding
                f.pop("sequence")
                f.pop("masked_sequence")

            # 2) Call tokenizer.pad(...) for proper batching
            batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            return batch

    def compute_cdr_indices(seq_unmasked: str, seq_masked: str) -> list:
        """
        Identify the indices where 'masked_sequence' has '<mask>' => CDR positions.
        Assumes seq_unmasked and seq_masked are aligned 1-to-1 at each index.
        """
        cdr_positions = []
        for idx, (u_char, m_char) in enumerate(zip(seq_unmasked, seq_masked)):
            if m_char == "<mask>":
                cdr_positions.append(idx)
        return cdr_positions

    print(f"\n🚀 Loading ESM model for MLM from: {checkpoint}")
    model_mlm = AutoModelForMaskedLM.from_pretrained(checkpoint)
    print("✅ Model loaded successfully.")

    # ---------------------------
    #  ADDING LR SCHEDULER, WARMUP, AND EARLY STOPPING
    # ---------------------------

    training_args = TrainingArguments(
        output_dir="/opt/ml/model/esm_mlm_cdr_pretraining",  # output to /opt/ml/model
        evaluation_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",        # Save model every epoch
        fp16=torch.cuda.is_available(),
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=50,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        optim="adamw_torch",
        remove_unused_columns=False,
        
        # >>> WARMUP & LR SCHEDULER:
        lr_scheduler_type="linear",
        warmup_ratio=0.1,  # 10% warmup

        # >>> EARLY STOPPING REQUIRES EVAL
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    print(f"✅ Training Arguments Configured: {training_args}")

    cdr_data_collator = DataCollatorCDRMasking(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )

    # >>> EARLY STOPPING CALLBACK
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0
    )

    trainer = Trainer(
        model=model_mlm,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=cdr_data_collator,
        callbacks=[early_stopping_callback],
    )

    print("✅ Trainer initialised successfully.")

    # 🚀 **Start Dynamic Masking Pre-training**
    print("\n🚀 Starting MLM Pre-training on masked CDR sequences (dynamic approach)...")
    trainer.train()
    print("\n✅ Pre-training complete!")

    # Sanitize the checkpoint name for use in the directory name
    safe_checkpoint = checkpoint.replace("/", "_")
    output_dir = f"/opt/ml/model/esm_mlm_cdr_pretrained_{safe_checkpoint}"
    
    print("\n💾 Saving pre-trained model to", output_dir, "...")
    model_mlm.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Pre-trained model saved successfully!")


if __name__ == "__main__":
    main()
