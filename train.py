import kagglehub
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize

print(" Training started...")

# Download dataset
path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
print("Dataset path:", path)

# Load dataset
df = pd.read_csv(os.path.join(path, "train.csv"))

# Rename columns
df.columns = ['label', 'title', 'description']

# Combine text columns
df['text'] = df['title'] + " " + df['description']

# Features and target
X = df['text']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression(max_iter=1000)
)

# Train model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# -------------------------
# Performance Metrics
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n📊 Model Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# -------------------------
# Cross Validation
# -------------------------
cv_scores = cross_val_score(pipeline, X, y, cv=5)

print("\n🔁 CV Scores:", cv_scores)
print("Average CV:", cv_scores.mean())

# -------------------------
# Save model
# -------------------------
pickle.dump(pipeline, open("model.pkl", "wb"))

print("\n✅ Model saved successfully!")

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

# -------------------------
# ROC Curve (Multiclass)
# -------------------------
classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)

# Probability scores
y_prob = pipeline.predict_proba(X_test)

plt.figure(figsize=(8,6))

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label=f'Class {classes[i]} (AUC={roc_auc:.2f})')

plt.plot([0,1], [0,1], '--')
plt.title("ROC Curve (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()