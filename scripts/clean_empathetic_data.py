import pandas as pd
import os
import csv

def clean_and_save_empathetic(input_path: str, output_path: str):
    df = pd.read_csv(input_path, quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')

    # Keep only utterance and emotion context
    df = df[["utterance", "context"]].rename(columns={"utterance": "text", "context": "label"})

    # Drop missing values
    df.dropna(subset=["text", "label"], inplace=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned and saved {output_path}")

if __name__ == "__main__":
    clean_and_save_empathetic(".\\raw_data\\empathetic\\empathetic_raw_train.csv", ".\\cleaned_data\\empathetic\\empathetic_train_cleaned.csv")
    clean_and_save_empathetic(".\\raw_data\\empathetic\\empathetic_raw_valid.csv", ".\\cleaned_data\\empathetic\\empathetic_valid_cleaned.csv")
    clean_and_save_empathetic(".\\raw_data\\empathetic\\empathetic_raw_test.csv", ".\\cleaned_data\\empathetic\\empathetic_test_cleaned.csv")
