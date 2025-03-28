import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned Reddit and Empathetic Dialogues datasets
reddit_path = '.\\cleaned_data\\reddit_cleaned.csv'
empathetic_path = '.\\cleaned_data\\empathetic_train_cleaned.csv'

# Load datasets
reddit_df = pd.read_csv(reddit_path)
empathetic_df = pd.read_csv(empathetic_path)

# Count label distribution
reddit_counts = reddit_df['label'].value_counts().sort_index()
empathetic_counts = empathetic_df['label'].value_counts().sort_values(ascending=False)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Reddit plot
reddit_counts.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Reddit Mental Health Label Distribution')
axes[0].set_xlabel('Label (0 = Stress ... 4 = Anxiety)')
axes[0].set_ylabel('Frequency')

# Empathetic plot
empathetic_counts.plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title('Empathetic Dialogues Emotion Distribution')
axes[1].set_xlabel('Emotion Label')
axes[1].set_ylabel('Frequency')
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()
