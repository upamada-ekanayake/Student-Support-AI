import streamlit as st
import joblib

# 1. Load the "Brain" and "Translator"
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 2. Define the Advice Logic (The same function we wrote in Colab)
def get_advice(emotion):
    advice_dict = {
        "Joy": "That is great! Keep up the positive energy.",
        "Sadness": "I am sorry you feel this way. Try talking to a friend or taking a short walk.",
        "Anger": "Take a deep breath. Count to 10. It is okay to feel frustrated.",
        "Fear": "You are safe. Focus on the present moment. One step at a time.",
        "Love": "That is a beautiful feeling. Share it with someone!",
        "Surprise": "Life is full of surprises! Embrace the moment."
    }
    return advice_dict.get(emotion, "I am here to listen.")

# 3. Build the Website Interface
st.title("🤖 Student Support AI")
st.write("Tell me how you are feeling, and I will try to help.")

# Input box for the user
user_input = st.text_area("Type your problem here:", height=100)

# Button to trigger the AI
if st.button("Analyze Emotion"):
    if user_input:
        # A. Translate text to numbers
        user_input_vec = vectorizer.transform([user_input])
        
        # B. Predict the emotion
        prediction = model.predict(user_input_vec)[0]
        
        # C. Get the label name (Translation)
        emotion_names = {
            0: "Sadness", 1: "Joy", 2: "Love", 
            3: "Anger", 4: "Fear", 5: "Surprise"
        }
        predicted_label = emotion_names.get(prediction, "Unknown")
        
        # D. Get Advice
        advice = get_advice(predicted_label)
        
        # E. Show the result
        st.success(f"Detected Emotion: **{predicted_label}**")
        st.info(f"💡 Suggestion: {advice}")
        
    else:
        st.warning("Please type something first!")
