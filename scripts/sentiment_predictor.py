import joblib
import os

# Load model + vectorizer
model_path = ".\\models\\reddit\\reddit_model.pkl"
vectorizer_path = ".\\models\\reddit\\reddit_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_condition(text: str) -> str:
    """Predict the mental health condition from user input"""
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    return prediction

# ✅ Optional: test run
if __name__ == "__main__":
    sample = input("Enter a message: ")
    result = predict_condition(sample)
    print(f"🧠 Predicted condition: {result}")
