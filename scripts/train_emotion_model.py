import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import torch
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)


def train_emotion_model(model_name: str, train_path: str, val_path: str, output_dir: str):
    # Load and encode labels
    print("✅ Using device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU only")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    le = LabelEncoder()
    train_df["labels"] = le.fit_transform(train_df["label"])
    val_df["labels"] = le.transform(val_df["label"])

    # Save label map
    os.makedirs(output_dir, exist_ok=True)
    pd.Series(le.classes_).to_csv(os.path.join(output_dir, "label_map.csv"), index_label="index")

    # Convert to HF datasets
    train_ds = Dataset.from_pandas(train_df[["text", "labels"]])
    val_ds = Dataset.from_pandas(val_df[["text", "labels"]])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize(example):
        return tokenizer(example["text"], truncation=True)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(le.classes_)
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": (preds == labels).mean()}

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved model: {model_name} to {output_dir}")


def get_model_name(model: str):
    models = {
                "distilbert": "distilbert-base-uncased",
                "bert-base": "bert-base-uncased",
                "bert-large": "bert-large-uncased",
                "bert-base-cased": "bert-base-cased",
                "bert-large-cased": "bert-large-cased",
                "roberta-base": "roberta-base",
                "roberta-large": "roberta-large",
                "albert": "albert-base-v2",
                "electra-base":	"google/electra-base-discriminator",
                "electra-large": "google/electra-large-discriminator",
                "deberta-base": "microsoft/deberta-v3-base",
                "deberta-large": "microsoft/deberta-v3-large",
                "xlnet-base": "xlnet-base-cased",
                "xlnet-large": "xlnet-large-cased"
                # Not utilising the following models as they are intended for multilingual data or high performance, which is outside of the scope of this project.
                # "camembert": "camembert-base",
                # "xlm-roberta-base": "xlm-roberta-base",
                # "xlm-roberta-large": "xlm-roberta-large",
                # "flaubert": "flaubert-base-cased",
                # "bart-base": "facebook/bart-base",
                # "bart-large": "facebook/bart-large",
                # "longformer-base": "allenai/longformer-base-4096"
            }
    if (model.lower() not in models.keys()) and (model.lower().split('-')[0] not in [i.split('-')[0] for i in models.keys()]):
        raise ValueError(f"Invalid model name: '{model}'. Please choose one of: {list(models.keys())}")
    else:
        if model.lower() in models.keys():
            model_name = models[model.lower()]
        else:
            model_name = models[[i for i in models.keys() if i.startswith(model)][0]]
            print(f"No model size provided. Using default: {model_name}")
    return model_name


def get_output_dir(model_name: str):
    output_dir_model_name = model_name.replace("/", "-").split('-')
    if len(output_dir_model_name) > 3:
        output_dir_model_name_1 = "_".join(output_dir_model_name[:2])
    elif len(output_dir_model_name) > 2 and (output_dir_model_name[2] == "base" or output_dir_model_name[2] == "large"):
        output_dir_model_name_1 = "_".join(output_dir_model_name[:2])
    else:
        output_dir_model_name_1 = "_".join(output_dir_model_name[:1])
    output_dir_model_name_2 = "_".join(output_dir_model_name)
    output_dir = f".\\models\\empathetic\\{output_dir_model_name_1}\\emotion_{output_dir_model_name_2}_finetuned"
    return output_dir


def train(model: str = "distilbert"):
    if len(sys.argv) > 1:
        model = str(sys.argv[1])
    else:
        print(f"No model name provided as CLI. Using default: {model}")
    model_name = get_model_name(model)
    train_path = ".\\cleaned_data\\empathetic_train_cleaned.csv"
    val_path = ".\\cleaned_data\\empathetic_valid_cleaned.csv"
    output_dir = get_output_dir(model_name)
    train_emotion_model (
                            model_name = model_name,
                            train_path = train_path,
                            val_path = val_path,
                            output_dir = output_dir
                        )

if __name__ == "__main__":
    train()
