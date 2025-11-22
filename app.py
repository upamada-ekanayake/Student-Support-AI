import streamlit as st
import joblib

# --- 1. LOAD ALL BRAINS ---
# Emotion Brain
emotion_model = joblib.load("model.pkl")
emotion_vectorizer = joblib.load("vectorizer.pkl")

# Topic Brain
topic_model = joblib.load("topic_model.pkl")
topic_vectorizer = joblib.load("topic_vectorizer.pkl")

# --- 2. INTELLIGENT ADVICE SYSTEM ---
# This dictionary combines [Topic] + [Emotion] to give specific advice
def get_personalized_advice(topic, emotion):
    advice_key = f"{topic}_{emotion}"
    
    strategies = {
        # ACADEMIC PROBLEMS
        "Academic_Sadness": "It's okay to feel disappointed. Analyze what went wrong, maybe change your study method, and try again. You are more than your grades.",
        "Academic_Fear": "Exam stress is real. Break your study material into tiny chunks. Focus on just one hour at a time.",
        "Academic_Anger": "Frustration is part of learning. Take a break, walk away from the books for 20 minutes, then restart.",
        
        # RELATIONSHIP PROBLEMS
        "Relationship_Sadness": "Heartbreak is painful. Give yourself time to grieve. Don't text them right now. Focus on self-care.",
        "Relationship_Fear": "Conflict is scary, but honest communication helps. Write down what you want to say before talking to them.",
        "Relationship_Anger": "Don't react in the heat of the moment. Write a letter to them but DON'T send it. Burn it to release the anger.",
        
        # CAREER PROBLEMS
        "Career_Sadness": "Rejection is not failure; it's redirection. Ask for feedback and keep improving your skills.",
        "Career_Fear": "The future is uncertain for everyone. Focus on building one small skill today (like updating your CV).",
        
        # MENTAL HEALTH
        "Mental Health_Sadness": "Please be gentle with yourself. You don't have to be productive today. Just drinking water and resting is enough.",
        "Mental Health_Fear": "Anxiety is lying to you. Try the 5-4-3-2-1 grounding technique: Name 5 things you see, 4 you feel, etc."
    }
    
    # If we don't have a specific combo, give a general fallback
    return strategies.get(advice_key, "I hear you. This sounds like a tough situation. Taking a deep breath is a good first step.")

# --- 3. THE WEBSITE UI ---
st.title("🧠 Student Support AI (Advanced)")
st.write("I can analyze your **Emotion** AND your **Problem Type**.")

user_input = st.text_area("Tell me what's wrong:", height=100)

if st.button("Get Personalized Help"):
    if user_input:
        # 1. Analyze Emotion
        em_vec = emotion_vectorizer.transform([user_input])
        current_emotion = emotion_names = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}[emotion_model.predict(em_vec)[0]]
        
        # 2. Analyze Topic
        top_vec = topic_vectorizer.transform([user_input])
        current_topic = topic_model.predict(top_vec)[0]
        
        # 3. Get Advice
        final_advice = get_personalized_advice(current_topic, current_emotion)
        
        # 4. Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Emotion:** {current_emotion}")
        with col2:
            st.info(f"**Topic:** {current_topic}")
            
        st.success(f"💡 **Personalized Solution:** {final_advice}")
        
    else:
        st.warning("Please type something.")
