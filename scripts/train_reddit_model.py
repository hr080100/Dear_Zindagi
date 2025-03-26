import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_reddit_model(input_path: str, model_path: str, vectorizer_path: str):
    # Load cleaned Reddit data
    df = pd.read_csv(input_path)
    X = df["text"]
    y = df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_vec)
    print("\n📊 Reddit Model Performance:\n")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    train_reddit_model(
        input_path=".\\cleaned_data\\reddit_cleaned.csv",
        model_path=".\\models\\reddit\\reddit_model.pkl",
        vectorizer_path=".\\models\\reddit\\reddit_vectorizer.pkl"
    )
