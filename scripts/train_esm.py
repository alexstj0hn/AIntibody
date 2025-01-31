# train.py
import os
import argparse

from datasets import load_dataset
from transformers import (
    EsmForMaskedLM,
    EsmTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

import shutil
shutil.rmtree("/opt/ml/model/checkpoint-*", ignore_errors=True)

import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))

def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--train_file", type=str, default="/opt/ml/input/data/train/train.csv")
    parser.add_argument("--validation_file", type=str, default="/opt/ml/input/data/validation/val.csv")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load dataset
    data_files = {}
    if os.path.isfile(args.train_file):
        data_files["train"] = args.train_file
    if os.path.isfile(args.validation_file):
        data_files["validation"] = args.validation_file

    raw_datasets = load_dataset("csv", data_files=data_files)

    # 2. Load tokenizer and model
    tokenizer = EsmTokenizer.from_pretrained(args.model_name)
    model = EsmForMaskedLM.from_pretrained(args.model_name)

    # 3. Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=1024
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["sequence", "id"] if "id" in raw_datasets["train"].column_names else ["sequence"]
    )

    # 4. Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # 5. TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="epoch" if "validation" in raw_datasets else "no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        report_to="none",
    )

    # 6. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if "train" in tokenized_datasets else None,
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
        data_collator=data_collator,
    )

    # 7. Train
    trainer.train()

    # 8. Save final model (SageMaker will automatically upload /opt/ml/model)
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()