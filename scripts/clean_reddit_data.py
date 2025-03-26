import pandas as pd
import os

LABEL_MAP = {
    0: "stress",
    1: "depression",
    2: "bipolar",
    3: "personality_disorder",
    4: "anxiety"
}

def clean_reddit_data(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Combine title + text
    df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["label"] = df["target"].map(LABEL_MAP)

    # Select final columns
    cleaned = df[["combined_text", "label"]].rename(columns={"combined_text": "text"})

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    clean_reddit_data(
        input_path = ".\\raw_data\\reddit_raw_data.csv",
        output_path = ".\\cleaned_data\\reddit_cleaned.csv"
    )
