import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

print("🚀 Starting analysis...")

# =====================================
# LOAD RAW DATASET
# =====================================

path = kagglehub.dataset_download(
    "amananandrai/ag-news-classification-dataset"
)

print("Dataset path:", path)

raw_df = pd.read_csv(os.path.join(path, "train.csv"))

# Rename columns
raw_df.columns = ['Category', 'Title', 'Description']

# =====================================
# BEFORE PREPROCESSING
# =====================================

print("📊 Showing BEFORE preprocessing graphs...")

# Create raw text
raw_df['Text'] = raw_df['Title'] + " " + raw_df['Description']

# Text length
raw_df['Text_Length'] = raw_df['Text'].apply(len)

# 1. Histogram (Raw Text Length)
plt.figure(figsize=(8,5))
sns.histplot(raw_df['Text_Length'], bins=30)
plt.title("Before Preprocessing - Text Length Distribution")
plt.tight_layout()
plt.show()

# 2. Boxplot (Raw Text Length)
plt.figure(figsize=(8,4))
sns.boxplot(x=raw_df['Text_Length'])
plt.title("Before Preprocessing - Text Length Outliers")
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    raw_df[['Category', 'Text_Length']].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Before Preprocessing - Correlation Matrix")
plt.tight_layout()
plt.show()

# =====================================
# PREPROCESSING
# =====================================

df = raw_df.copy()

# Label Encoding
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# Feature Engineering
df['Title_Length'] = df['Title'].apply(len)
df['Description_Length'] = df['Description'].apply(len)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000
)

X_tfidf = vectorizer.fit_transform(df['Text'])

# =====================================
# AFTER PREPROCESSING
# =====================================

print("📊 Showing AFTER preprocessing graphs...")

# 1. Histogram
plt.figure(figsize=(8,5))
sns.histplot(df['Text_Length'], bins=30, kde=True)
plt.title("After Preprocessing - Text Length Distribution")
plt.tight_layout()
plt.show()

# 2. Boxplot
plt.figure(figsize=(8,4))
sns.boxplot(x=df['Text_Length'])
plt.title("After Preprocessing - Text Length Outliers")
plt.tight_layout()
plt.show()

# 3. Correlation Matrix
plt.figure(figsize=(8,6))

sns.heatmap(
    df[['Category',
        'Text_Length',
        'Title_Length',
        'Description_Length']].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("After Preprocessing - Correlation Matrix")
plt.tight_layout()
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(8,5))

sns.scatterplot(
    x=df['Title_Length'],
    y=df['Text_Length']
)

plt.title("Title Length vs Text Length")
plt.tight_layout()
plt.show()

# =====================================
# FEATURE IMPORTANCE
# =====================================

print("📈 Calculating feature importance...")

X = pd.DataFrame({
    'Text_Length': df['Text_Length'],
    'Title_Length': df['Title_Length'],
    'Description_Length': df['Description_Length']
})

y = df['Category']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

importance = model.feature_importances_

# Feature Importance Plot
plt.figure(figsize=(8,5))

plt.bar(X.columns, importance)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()
plt.show()

print("✅ Analysis complete!")



