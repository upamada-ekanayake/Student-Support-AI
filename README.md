# 🧠 Student Support AI (Advanced)

A Context-Aware Mental Health Assistant built with Python & Scikit-Learn.
**Live Demo:** [Click Here to Open App](https://share.streamlit.io/upamada-ekanayake/Student-Support-AI/main/app.py)

## 🚀 Project Overview
This application helps students manage stress by analyzing their raw text input. Unlike simple sentiment analyzers, this AI uses a **Double-Brain Architecture** to understand both **Emotion** and **Topic** simultaneously.

### 🛠️ How it Works
The system runs two Logistic Regression models in parallel:
1.  **Emotion Classifier:** Detects 6 emotions (Joy, Sadness, Anger, Fear, Love, Surprise). *Trained on 16,000 data points.*
2.  **Topic Classifier:** Detects the root cause (Academic, Relationship, Career, Mental Health).

Based on the combination (e.g., `Academic` + `Fear`), the system looks up a specialized psychological strategy to assist the user.

## 💻 Tech Stack
* **Language:** Python
* **ML Libraries:** Scikit-Learn, Pandas, Joblib
* **NLP:** TF-IDF Vectorization, Text Preprocessing
* **Web Framework:** Streamlit
* **Hosting:** Streamlit Cloud

## 📊 Key Features
* **Real-time NLP:** Instantly classifies text input.
* **Balanced Training:** Implemented Class Weights to handle imbalanced datasets (fixing the "Surprise" class recall).
* **Personalized Logic:** Maps (Emotion + Topic) pairs to specific actionable advice.

## ✍️ Author
**Upamada Ekanayake** - *AI & Data Science Student*
