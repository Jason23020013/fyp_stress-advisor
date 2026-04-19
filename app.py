import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import time
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# --- NEW: GEMINI INTEGRATION ---
try:
    import google.generativeai as genai
except ImportError:
    st.error("🚨 Critical Error: 'google-generativeai' library is missing. Please run: pip install google-generativeai")
    st.stop()

# --- DATABASE CONNECTION ---
def get_db_connection():
    # check_same_thread=False is crucial for Streamlit
    conn = sqlite3.connect('student_stress.db', check_same_thread=False)
    return conn

# --- CHECK FOR SMOTE ---
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    st.error("🚨 Critical Error: 'imbalanced-learn' is missing. Please run: pip install imbalanced-learn")
    st.stop()

# ==========================================
# 1. INTERNAL BRAIN (SQL Powered)
# ==========================================
@st.cache_resource
def train_internal_model(force_retrain=False):
    if not os.path.exists('student_stress.db'):
        return None, None, 0, 0, None, None

    conn = get_db_connection()
    
    # FETCH DATA FROM SQL
    query = """
    SELECT * FROM training_data
    UNION ALL
    SELECT Study_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, 
           Physical_Activity_Hours_Per_Day, Extracurricular_Hours_Per_Day, 
           GPA, Stress_Level 
    FROM user_feedback
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        conn.close()
        return None, None, 0, 0, None, None
    finally:
        conn.close()

    if df.empty:
         return None, None, 0, 0, None, None

    # Feature Engineering
    df['Lifestyle_Score'] = (df['Social_Hours_Per_Day'] + 
                             df['Physical_Activity_Hours_Per_Day'] + 
                             df['Extracurricular_Hours_Per_Day'])
    
    df['Academic_Pressure'] = df['GPA'] * df['Study_Hours_Per_Day']

    features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Lifestyle_Score', 'Academic_Pressure', 'GPA']
    X = df[features]
    y = df['Stress_Level']
    
    # Encode & Split
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    
    # Train
    model = RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_split=10, random_state=42)
    model.fit(X_bal, y_bal)
    
    # Calculate Accuracies
    train_pred = model.predict(X_bal)
    train_acc = accuracy_score(y_bal, train_pred)
    
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    cm = confusion_matrix(y_test, test_pred)
    
    return model, le, train_acc, test_acc, cm, features

# Load Model
model, le, train_acc, test_acc, model_cm, feature_names = train_internal_model()

# ==========================================
# 2. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Gemini AI Counselor", layout="wide")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3062/3062331.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🤖 AI Predictor", "💬 AI Chatbot", "📝 User Survey", "📈 Data Analysis", "📊 Dashboard"])

# ==========================================
# 3. API KEY SETUP (PERMANENT)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🔑 Connection Status")

api_key = st.secrets["GEMINI_API_KEY"]

if api_key == "" or "PASTE_YOUR_KEY" in api_key:
    gemini_model = None
    st.sidebar.error("🔴 No API Key Found")
    st.sidebar.info("Please paste your unique key from Google AI Studio.")
else:
    try:
        genai.configure(api_key=api_key)
        # UPDATED: Changed from gemini-1.5-flash to gemini-2.5-flash
        gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
        st.sidebar.success("🟢 AI Connected (Gemini 2.5 Flash)")
    except Exception as e:
        gemini_model = None
        st.sidebar.error(f"🔴 Connection Error: {e}")

# ==========================================
# 4. PAGE LOGIC
# ==========================================

# --- PAGE: HOME ---
if page == "🏠 Home":
    st.title("🧠 AI Student Stress Counselor")
    st.markdown("""
    Welcome to the **Next-Gen Student Well-being System**.
    
    **Features:**
    - **🤖 Predictive AI:** Uses Random Forest to calculate stress risk.
    - **💬 Generative AI:** A chatbot counselor powered by **Google Gemini**.
    - **🔄 Continuous Learning:** The system gets smarter with your feedback.
    """)
    if test_acc:
        col1, col2 = st.columns(2)
        col1.info(f"🎓 Training Accuracy: {train_acc*100:.1f}% (Learning Ability)")
        col2.success(f"🧪 Testing Accuracy: {test_acc*100:.1f}% (Real-world Performance)")

# --- PAGE: PREDICTOR ---
elif page == "🤖 AI Predictor":
    st.title("🤖 AI Stress Assessment")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Your Daily Habits")
        study = st.slider("Study Hours (per day)", 0.0, 12.0, 5.0)
        sleep = st.slider("Sleep Hours (per day)", 0.0, 12.0, 7.0)
        social = st.slider("Social Hours (per day)", 0.0, 10.0, 2.0)
        physical = st.slider("Physical Activity (per day)", 0.0, 5.0, 1.0)
        extra = st.slider("Extracurriculars (per day)", 0.0, 5.0, 1.0)
        
        # --- NEW CODE: 24-HOUR TRACKER ---
        total_hours = study + sleep + social + physical + extra
        hours_left = 24.0 - total_hours

        # Dynamic Visual Feedback
        if hours_left >= 0:
            st.markdown(f"**⏱️ Time Budget:** :green[{hours_left:.1f} hours remaining] (Used: {total_hours}/24)")
            st.progress(total_hours / 24.0)
        else:
            st.markdown(f"**🚨 Time Overload:** :red[You are over by {abs(hours_left):.1f} hours!] (Used: {total_hours}/24)")
            st.progress(1.0) # Full bar (red context via text)
        # ---------------------------------

        gpa = st.slider("Current GPA", 0.0, 4.0, 3.0)
        
        lifestyle_score = social + physical + extra
        academic_pressure = gpa * study
        
        # Validation Logic
        if total_hours > 24:
            st.error(f"🚨 Invalid Input: Total hours ({total_hours}) cannot exceed 24.")
        else:
            if st.button("Analyze Stress Level"):
                if model:
                    input_data = pd.DataFrame([[study, sleep, lifestyle_score, academic_pressure, gpa]], columns=feature_names)
                    
                    # Prediction
                    pred_idx = model.predict(input_data)[0]
                    pred_proba = model.predict_proba(input_data)[0]
                    confidence = np.max(pred_proba) * 100
                    pred_label = le.inverse_transform([pred_idx])[0]
                    
                    # --- GENERATE GEMINI ADVICE ---
                    ai_advice = ""
                    if gemini_model:
                        with st.spinner("🤖 Consulting Gemini for personalized advice..."):
                            try:
                                prompt = f"""
Act as a professional and empathetic University Counselor. 
A student has the following profile:
- Study Hours: {study} hrs/day
- Sleep Hours: {sleep} hrs/day
- GPA: {gpa}
- AI Predicted Stress Level: {pred_label} (Confidence: {confidence:.1f}%)

Please provide:
1. A brief 1-sentence analysis of why these habits might lead to this stress level.
2. Three (3) specific, actionable tips tailored to this student's data to help them improve their well-being.
Keep the tone supportive, professional, and concise.
"""
                                response = gemini_model.generate_content(prompt)
                                ai_advice = response.text
                            except Exception as e:
                                ai_advice = "Error connecting to Gemini. Check API Key."
                    
                    st.session_state['last_pred'] = {
                        'res': pred_label, 
                        'conf': confidence,
                        'advice': ai_advice,
                        'inputs': (study, sleep, social, physical, extra, gpa, pred_label),
                        'sleep': sleep, 'study': study
                    }
                    st.session_state['feedback_mode'] = False 

    with c2:
        st.subheader("Analysis Result")
        if 'last_pred' in st.session_state:
            res = st.session_state['last_pred']['res']
            conf = st.session_state['last_pred']['conf']
            advice = st.session_state['last_pred'].get('advice', "")
            p = st.session_state['last_pred']
            
            # Gauge Chart
            color_map = {"Low": "green", "Moderate": "orange", "High": "red"}
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = conf,
                title = {'text': f"Prediction: {res} Stress"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': color_map.get(res, "blue")},
                         'steps' : [{'range': [0, 50], 'color': "lightgray"}, {'range': [50, 100], 'color': "gray"}]}))
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"🤖 AI Confidence: **{conf:.1f}%**")

            # --- DYNAMIC RECOMMENDATIONS ---
            st.markdown("### 💡 AI Recommendations:")
            if advice and "Error" not in advice:
                st.success(advice) # Show Gemini Response
            else:
                # Fallback Logic
                if res == "High": 
                    st.write("- 🛑 **Immediate Action:** Reduce study intensity.")
                    st.write("- 🛌 **Sleep:** Try to get at least 7 hours.")
                elif res == "Moderate": 
                    st.write("- ⚖️ **Balance:** Schedule one social activity.")
                else: 
                    st.write("- ✨ **Great Job:** Keep up this healthy routine!")
            
            st.markdown("---")
            st.write("Is this result correct?")
            c_yes, c_no = st.columns(2)
            
            if c_yes.button("✅ Yes, Correct"):
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("""INSERT INTO user_feedback (Study_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, Physical_Activity_Hours_Per_Day, Extracurricular_Hours_Per_Day, GPA, Stress_Level) VALUES (?, ?, ?, ?, ?, ?, ?)""", p['inputs'])
                conn.commit()
                conn.close()
                st.success("Feedback saved!")

            if c_no.button("❌ No, It's Wrong"):
                st.session_state['feedback_mode'] = True

            # --- CORRECTION LOGIC ---
            if st.session_state.get('feedback_mode', False):
                with st.form("correction_form"):
                    st.warning("Please tell us the correct stress level:")
                    correct_label = st.selectbox("Correct Stress Level", ["Low", "Moderate", "High"])
                    if st.form_submit_button("Submit Correction"):
                        conn = get_db_connection()
                        cur = conn.cursor()
                        inputs = list(p['inputs']) 
                        inputs[-1] = correct_label 
                        cur.execute("""INSERT INTO user_feedback (Study_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, Physical_Activity_Hours_Per_Day, Extracurricular_Hours_Per_Day, GPA, Stress_Level) VALUES (?, ?, ?, ?, ?, ?, ?)""", tuple(inputs))
                        conn.commit()
                        conn.close()
                        st.success("Correction Saved!")
                        st.session_state['feedback_mode'] = False
                        st.rerun()

# --- PAGE: CHATBOT (GEMINI) ---
elif page == "💬 AI Chatbot":
    st.title("💬 Gemini Student Counselor")
    st.markdown("Feel free to discuss your stress or academic pressure here.")

    if not gemini_model:
        st.warning("⚠️ Gemini is disconnected. Check the sidebar status.")
    else:
        # Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("How are you feeling today?"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI Response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    # Construct Prompt from history
                    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    system_prompt = """
You are an empathetic, supportive, and professional AI Student Counselor. 
Your goal is to listen to students' academic and personal concerns. 
Provide helpful, encouraging, and non-judgmental feedback. 
Keep your responses warm but concise. 
If a student expresses high levels of stress, gently remind them of the importance of self-care and seeking balance.
\n\n
"""
                    
                    response = gemini_model.generate_content(system_prompt + history_text)
                    full_response = response.text
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    message_placeholder.error(f"Error: {e}")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- PAGE: SURVEY ---
elif page == "📝 User Survey":
    st.title("📝 Student Lifestyle Survey")
    with st.form("survey_form"):
        s_study = st.number_input("Study Hours", 0.0, 24.0, 5.0)
        s_sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
        s_social = st.number_input("Social Hours", 0.0, 24.0, 2.0)
        s_phys = st.number_input("Physical Activity", 0.0, 24.0, 1.0)
        s_extra = st.number_input("Extracurriculars", 0.0, 24.0, 1.0)
        s_gpa = st.number_input("GPA", 0.0, 4.0, 3.0)
        s_stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
        if st.form_submit_button("Submit Survey"):
            if s_study + s_sleep + s_social + s_phys + s_extra > 24:
                st.error("Error: > 24 Hours")
            else:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("""INSERT INTO user_feedback (Study_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, Physical_Activity_Hours_Per_Day, Extracurricular_Hours_Per_Day, GPA, Stress_Level) VALUES (?, ?, ?, ?, ?, ?, ?)""", (s_study, s_sleep, s_social, s_phys, s_extra, s_gpa, s_stress))
                conn.commit()
                conn.close()
                st.success("Data added.")

# --- PAGE: DATA ANALYSIS ---
elif page == "📈 Data Analysis":
    st.title("📈 Model Transparency")
    t1, t2, t3 = st.tabs(["Dataset", "Feature Importance", "Performance"])
    with t1:
        conn = get_db_connection()
        try:
            df_display = pd.read_sql("SELECT * FROM training_data LIMIT 100", conn)
            fig = px.box(df_display, x="Stress_Level", y="Sleep_Hours_Per_Day", color="Stress_Level")
            st.plotly_chart(fig)
        except:
            st.write("No data found.")
        conn.close()
        
    with t2:
        if model:
            imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
            st.plotly_chart(px.bar(imp, x='Importance', y='Feature', orientation='h'))
    with t3:
        if test_acc: st.metric("Model Accuracy", f"{test_acc*100:.2f}%")
        if model_cm is not None: st.plotly_chart(px.imshow(model_cm, text_auto=True, x=le.classes_, y=le.classes_, color_continuous_scale='Blues'))

# --- PAGE: DASHBOARD ---
elif page == "📊 Dashboard":
    st.title("Admin Dashboard & System Testing")
    conn = get_db_connection()
    try: count_feed = pd.read_sql("SELECT COUNT(*) as count FROM user_feedback", conn)['count'][0]
    except: count_feed = 0
    conn.close()
    col1, col2 = st.columns(2)
    col1.metric("Testing Accuracy", f"{test_acc*100:.1f}%")
    col2.metric("New Feedback", count_feed)
    st.markdown("---")
    
    st.subheader("⚙️ Model Maintenance")
    if st.button("🔄 Retrain Model"):
        if count_feed == 0: st.warning("No new data.")
        else:
            start_time = time.time()
            with st.spinner("Retraining..."):
                st.cache_resource.clear()
                model, le, tr_acc, te_acc, cm, feats = train_internal_model(force_retrain=True)
            end_time = time.time()
            st.success(f"Retrained! Time: {end_time - start_time:.4f}s | New Acc: {te_acc*100:.2f}%")