import streamlit as st
import pickle
import requests

# Page config
st.set_page_config(page_title="News Classifier", layout="wide")

# 🌸 Custom CSS (FINAL)
st.markdown("""
<style>
.stApp {
    background-color: #ffe4ec;
    color: #87CEEB; /* Sky blue text */
}

/* 🔥 Deep teal title */
h1 {
    color: #006d6f !important;
}

/* 🔥 Dark pink subheading */
h3 {      
    color: #d63384 !important;
}

/* Butter yellow cards */
.card {
    background-color: #fff4cc;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    color: #4682B4;
}

/* 🔥 Dark pink button */
button {
    background-color: #d63384;
    color: white;
    border: none;
    padding: 10px;
    border-radius: 10px;
    cursor: pointer;
}

/* 🔥 Button hover */
button:hover {
    background-color: #b52a6b;
}

/* Flower animation */
.flower {
    position: fixed;
    top: -50px;
    font-size: 24px;
    animation: fall linear infinite;
}

@keyframes fall {
    to {
        transform: translateY(100vh);
    }
}
</style>

<!-- 🌸 Floating flowers -->
<div class="flower" style="left:10%; animation-duration:8s;">🌸</div>
<div class="flower" style="left:30%; animation-duration:10s;">🌸</div>
<div class="flower" style="left:50%; animation-duration:12s;">🌸</div>
<div class="flower" style="left:70%; animation-duration:9s;">🌸</div>
<div class="flower" style="left:90%; animation-duration:11s;">🌸</div>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("📰 Automated News Classification System")

# API key
API_KEY = "7717c15bb55349828c2e2dd292397254"

# Categories
categories = {
    1: "World 🌍",
    2: "Sports ⚽",
    3: "Business 💼",
    4: "Sci/Tech 💻"
}

# Sidebar filter
st.sidebar.header("🔎 Filter Categories")
selected_categories = st.sidebar.multiselect(
    "Choose categories",
    list(categories.values()),
    default=list(categories.values())
)

# Fetch news
all_articles = []

for page in range(1, 3):
    url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=20&page={page}&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        all_articles.extend(data["articles"])

# Heading
st.subheader("📡 Live News Feed")

# Layout
col1, col2 = st.columns(2)
i = 0

# Display news
for article in all_articles[:40]:
    news = article.get("title")
    link = article.get("url")

    if news:
        prediction = model.predict([news])[0]
        confidence = model.predict_proba([news]).max()
        category = categories.get(prediction)

        if category not in selected_categories:
            continue

        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="card">
                <h4>📰 {news}</h4>
                <p><b>Category:</b> {category}</p>
                <p><b>Confidence:</b> {confidence:.2f}</p>
                <a href="{link}" target="_blank">
                    <button>🔗 Read Full Article</button>
                </a>
            </div>
            """, unsafe_allow_html=True)

        i += 1

# Error handling
if not all_articles:
    st.error("❌ Failed to fetch news. Check API key or internet.")
