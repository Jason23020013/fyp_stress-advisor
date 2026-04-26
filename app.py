import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import time
import sqlite3
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. Core Library Imports & Checks ---
try:
    from supabase import create_client, Client
    import google.generativeai as genai
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    st.error(f"🚨 Missing required libraries: {e}. Please ensure your requirements.txt includes supabase, google-generativeai, and imbalanced-learn.")
    st.stop()

# ==========================================
# 0. Cloud Database & Authentication Setup
# ==========================================

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error("❌ Failed to connect to the Supabase Cloud Database. Please check your Streamlit Secrets configuration.")
    st.stop()

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = "Guest"
if 'feedback_mode' not in st.session_state:
    st.session_state['feedback_mode'] = False
if 'chat_loaded' not in st.session_state:
    st.session_state['chat_loaded'] = False

# ==========================================
# 1. Login, Register, Forgot Password & GUEST
# ==========================================
if not st.session_state['logged_in']:
    st.set_page_config(page_title="🔐 System Access", layout="centered")
    st.title("🎓 Student Wellness & Stress Advisor")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Login", "Register New Account", "Forgot Password"])
    
    with tab1:
        st.subheader("Account Login")
        l_user = st.text_input("Student ID", placeholder="e.g., STU123456", key="login_u")
        l_pwd = st.text_input("Password", type="password", key="login_p")
        if st.button("Sign In", use_container_width=True):
            res = supabase.table("users").select("*").eq("student_id", l_user).execute()
            if res.data and check_hashes(l_pwd, res.data[0]['password_hash']):
                st.session_state['logged_in'] = True
                st.session_state['username'] = l_user
                st.session_state['user_role'] = res.data[0].get('role', 'Student')
                st.session_state['messages'] = [] 
                st.session_state['chat_loaded'] = False
                st.session_state.pop('last_pred', None) 
                st.success(f"Welcome back, {l_user}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid Student ID or Password. Please try again.")
    
    with tab2:
        st.info("Note: Please remember your Recovery Word. You will need it if you forget your password.")
        r_user = st.text_input("Set Student ID", placeholder="e.g., STU123456", key="reg_u")
        r_pwd = st.text_input("Set Password", type="password", key="reg_p")
        r_rec = st.text_input("Set Recovery Word (e.g., your pet's name)", type="password", key="reg_rec")
        
        if st.button("Register Now", use_container_width=True):
            if r_user and r_pwd and r_rec:
                hashed_pwd = make_hashes(r_pwd)
                hashed_rec = make_hashes(r_rec)
                try:
                    supabase.table("users").insert({
                        "student_id": r_user, 
                        "password_hash": hashed_pwd, 
                        "recovery_word_hash": hashed_rec,
                        "role": "Student"
                    }).execute()
                    st.success("Registration successful! Please switch to the Login tab.")
                except Exception as e:
                    st.error("Registration failed. This Student ID might already exist.")
            else:
                st.error("Please fill in all fields (ID, Password, and Recovery Word).")
                
    with tab3:
        st.subheader("Recover Your Account")
        st.write("Use your Security Recovery Word to set a new password.")
        f_user = st.text_input("Your Student ID", key="f_u")
        f_rec = st.text_input("Your Recovery Word", type="password", key="f_r")
        f_new = st.text_input("Enter New Password", type="password", key="f_n")
        
        if st.button("Reset Password", use_container_width=True):
            if f_user and f_rec and f_new:
                res = supabase.table("users").select("*").eq("student_id", f_user).execute()
                if res.data:
                    db_rec_hash = res.data[0].get('recovery_word_hash')
                    if db_rec_hash and check_hashes(f_rec, db_rec_hash):
                        new_pwd_hash = make_hashes(f_new)
                        supabase.table("users").update({"password_hash": new_pwd_hash}).eq("student_id", f_user).execute()
                        st.success("Password reset successfully! You can now log in.")
                    else:
                        st.error("Incorrect Recovery Word or no recovery word set for this account.")
                else:
                    st.error("Student ID not found.")
            else:
                st.error("Please fill in all fields.")
    
    st.markdown("---")
    st.subheader("Or explore the system directly")
    if st.button("🚀 Continue as Guest", use_container_width=True):
        st.session_state['logged_in'] = True
        st.session_state['username'] = "Guest User"
        st.session_state['user_role'] = "Guest"
        st.session_state['messages'] = [] 
        st.session_state['chat_loaded'] = False
        st.session_state.pop('last_pred', None) 
        st.rerun()
                
    st.stop()

# ==========================================
# 2. Main System Logic (Fully Cloud Integrated)
# ==========================================

@st.cache_resource
def train_internal_model(force_retrain=False):
    # 1. Load static base training data from local SQLite (if it exists)
    try:
        conn = sqlite3.connect('student_stress.db', check_same_thread=False)
        base_df = pd.read_sql_query("SELECT Study_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, Physical_Activity_Hours_Per_Day, Extracurricular_Hours_Per_Day, GPA, Stress_Level FROM training_data", conn)
        conn.close()
    except Exception:
        base_df = pd.DataFrame()

    # 2. Load dynamic user feedback securely from Supabase Cloud
    try:
        res = supabase.table("user_feedback").select("Study_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, Physical_Activity_Hours_Per_Day, Extracurricular_Hours_Per_Day, GPA, Stress_Level").execute()
        if res.data:
            feedback_df = pd.DataFrame(res.data)
        else:
            feedback_df = pd.DataFrame()
    except Exception:
        feedback_df = pd.DataFrame()

    # Combine data for training
    df = pd.concat([base_df, feedback_df], ignore_index=True)

    if df.empty:
         return None, None, 0, 0, None, None

    df['Lifestyle_Score'] = (df['Social_Hours_Per_Day'] + df['Physical_Activity_Hours_Per_Day'] + df['Extracurricular_Hours_Per_Day'])
    df['Academic_Pressure'] = df['GPA'] * df['Study_Hours_Per_Day']

    features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Lifestyle_Score', 'Academic_Pressure', 'GPA']
    X = df[features]
    y = df['Stress_Level']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_split=10, random_state=42)
    model.fit(X_bal, y_bal)
    
    train_pred = model.predict(X_bal)
    train_acc = accuracy_score(y_bal, train_pred)
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)
    
    return model, le, train_acc, test_acc, cm, features

model, le, train_acc, test_acc, model_cm, feature_names = train_internal_model()

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Gemini AI Counselor", layout="wide")

st.sidebar.markdown(f"### 👋 Welcome, {st.session_state['username']}")
st.sidebar.write(f"Current Role: `{st.session_state['user_role']}`")
if st.sidebar.button("🚪 Logout"):
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""
    st.session_state['messages'] = []
    st.session_state['chat_loaded'] = False
    st.session_state.pop('last_pred', None) 
    st.rerun()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3062/3062331.png", width=100)
st.sidebar.title("Navigation")

menu_options = ["🏠 Home", "🤖 AI Predictor", "💬 AI Chatbot"]
if st.session_state['user_role'] in ["Student", "Admin"]:
    menu_options += ["📜 My History", "📝 User Survey", "⚙️ Account Settings"]
if st.session_state['user_role'] == "Admin":
    menu_options += ["📈 Data Analysis", "📊 Dashboard"]

page = st.sidebar.radio("Go to", menu_options)

# --- API KEY SETUP ---
st.sidebar.markdown("---")
st.sidebar.header("🔑 AI Connection Status")
api_key = st.secrets["GEMINI_API_KEY"]

if api_key == "" or "PASTE_YOUR_KEY" in api_key:
    gemini_model = None
    st.sidebar.error("🔴 No API Key Found")
else:
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
        st.sidebar.success("🟢 AI Connected (Gemini 2.5)")
    except Exception as e:
        gemini_model = None
        st.sidebar.error(f"🔴 Connection Error: {e}")

# ==========================================
# 4. PAGE LOGIC
# ==========================================

if page == "🏠 Home":
    st.title("🧠 AI Student Stress Counselor")
    st.markdown("""
    Welcome to the **Next-Gen Student Well-being System**.
    
    **Features:**
    - **🤖 Predictive AI:** Uses Random Forest to calculate stress risk.
    - **💬 Generative AI:** A chatbot counselor powered by **Google Gemini**.
    - **📜 User History:** Securely tracks your past stress levels in the cloud.
    - **🔄 Continuous Learning:** The system gets smarter with your verified feedback.
    """)
    if test_acc:
        col1, col2 = st.columns(2)
        col1.info(f"🎓 Training Accuracy: {train_acc*100:.1f}% (Learning Ability)")
        col2.success(f"🧪 Testing Accuracy: {test_acc*100:.1f}% (Real-world Performance)")

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
        
        total_hours = study + sleep + social + physical + extra
        hours_left = 24.0 - total_hours

        if hours_left >= 0:
            st.markdown(f"**⏱️ Time Budget:** :green[{hours_left:.1f} hours remaining] (Used: {total_hours}/24)")
            st.progress(total_hours / 24.0)
        else:
            st.markdown(f"**🚨 Time Overload:** :red[You are over by {abs(hours_left):.1f} hours!] (Used: {total_hours}/24)")
            st.progress(1.0)

        gpa = st.slider("Current GPA", 0.0, 4.0, 3.0)
        lifestyle_score = social + physical + extra
        academic_pressure = gpa * study
        
        if total_hours > 24:
            st.error(f"🚨 Invalid Input: Total hours ({total_hours}) cannot exceed 24.")
        else:
            if st.button("Analyze Stress Level"):
                if model:
                    input_data = pd.DataFrame([[study, sleep, lifestyle_score, academic_pressure, gpa]], columns=feature_names)
                    pred_idx = model.predict(input_data)[0]
                    pred_proba = model.predict_proba(input_data)[0]
                    confidence = np.max(pred_proba) * 100
                    pred_label = le.inverse_transform([pred_idx])[0]
                    
                    # --- AUTO-SAVE TO CLOUD HISTORY ---
                    if st.session_state['user_role'] != "Guest":
                        try:
                            supabase.table("user_history").insert({
                                "student_id": st.session_state['username'],
                                "Study_Hours_Per_Day": study, "Sleep_Hours_Per_Day": sleep,
                                "Social_Hours_Per_Day": social, "Physical_Activity_Hours_Per_Day": physical,
                                "Extracurricular_Hours_Per_Day": extra, "GPA": gpa, "Stress_Level": pred_label
                            }).execute()
                        except Exception as e:
                            print(f"Error saving history: {e}")

                    ai_advice = ""
                    if gemini_model:
                        with st.spinner("🤖 Consulting Gemini for personalized advice..."):
                            try:
                                prompt = f"Act as a professional University Counselor. Student profile: Study {study} hrs/day, Sleep {sleep} hrs/day, GPA {gpa}. Predicted Stress: {pred_label} ({confidence:.1f}%). Provide a 1-sentence analysis and 3 specific actionable tips."
                                response = gemini_model.generate_content(prompt)
                                ai_advice = response.text
                            except Exception as e:
                                ai_advice = "Error connecting to Gemini. Check API Key."
                    
                    st.session_state['last_pred'] = {
                        'res': pred_label, 
                        'conf': confidence,
                        'advice': ai_advice,
                        'inputs': {"Study_Hours_Per_Day": study, "Sleep_Hours_Per_Day": sleep, "Social_Hours_Per_Day": social, "Physical_Activity_Hours_Per_Day": physical, "Extracurricular_Hours_Per_Day": extra, "GPA": gpa, "Stress_Level": pred_label}
                    }
                    st.session_state['feedback_mode'] = False 

    with c2:
        st.subheader("Analysis Result")
        if 'last_pred' in st.session_state:
            p = st.session_state['last_pred']
            color_map = {"Low": "green", "Moderate": "orange", "High": "red"}
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = p['conf'],
                title = {'text': f"Prediction: {p['res']} Stress"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': color_map.get(p['res'], "blue")},
                         'steps' : [{'range': [0, 50], 'color': "lightgray"}, {'range': [50, 100], 'color': "gray"}]}))
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"🤖 AI Confidence: **{p['conf']:.1f}%**")
            st.markdown("### 💡 AI Recommendations:")
            st.success(p['advice'])
            
            st.markdown("---")
            
            if st.session_state['user_role'] == "Guest":
                st.info("🔒 Log in to provide verified feedback and help improve our AI.")
            else:
                st.write("Does this prediction accurately describe you?")
                cy, cn = st.columns(2)
                
                if cy.button("✅ Yes, This is Accurate"):
                    feedback_data = p['inputs'].copy()
                    feedback_data["student_id"] = st.session_state['username']
                    try:
                        supabase.table("user_feedback").insert(feedback_data).execute()
                        st.success("Thanks! Saved to the cloud to improve the AI.")
                    except Exception as e:
                        st.error(f"Error saving to cloud: {e}")
                    
                if cn.button("❌ No, Let me correct it"):
                    st.session_state['feedback_mode'] = True

                if st.session_state.get('feedback_mode', False):
                    with st.form("correction_form"):
                        correct_label = st.selectbox("Your actual stress level", ["Low", "Moderate", "High"])
                        if st.form_submit_button("Submit Correction for AI Training"):
                            feedback_data = p['inputs'].copy()
                            feedback_data["Stress_Level"] = correct_label
                            feedback_data["student_id"] = st.session_state['username']
                            try:
                                supabase.table("user_feedback").insert(feedback_data).execute()
                                st.success("Correction saved to cloud for AI retraining!")
                            except Exception as e:
                                st.error(f"Error saving to cloud: {e}")
                            st.session_state['feedback_mode'] = False
                            st.rerun()

elif page == "💬 AI Chatbot":
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("💬 Gemini Student Counselor")
    with col2:
        if st.session_state['user_role'] != "Guest":
            if st.button("🗑️ Clear Chat"):
                supabase.table("user_chat_history").delete().eq("student_id", st.session_state['username']).execute()
                st.session_state.messages = []
                st.session_state['chat_loaded'] = True 
                st.rerun()
    
    if st.session_state['user_role'] == "Guest":
         st.info("👋 You are chatting as a Guest. Your chat history will not be saved after you log out.")
         if "messages" not in st.session_state:
             st.session_state.messages = []
    else:
        # LOAD CHAT FROM CLOUD DB
        if not st.session_state.get('chat_loaded', False):
            try:
                res = supabase.table("user_chat_history").select("role, content").eq("student_id", st.session_state['username']).order("created_at").execute()
                if res.data:
                    st.session_state.messages = [{"role": row['role'], "content": row['content']} for row in res.data]
                else:
                    st.session_state.messages = []
            except Exception as e:
                st.session_state.messages = []
            st.session_state['chat_loaded'] = True
         
    if not gemini_model:
        st.warning("⚠️ Gemini is disconnected.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): 
                st.markdown(message["content"])
                
        if prompt := st.chat_input("How are you feeling today?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            if st.session_state['user_role'] != "Guest":
                supabase.table("user_chat_history").insert({"student_id": st.session_state['username'], "role": "user", "content": prompt}).execute()
                
            with st.chat_message("user"): 
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                system_prompt = "You are an empathetic AI Student Counselor. Support the student warmly and concisely."
                response = gemini_model.generate_content(system_prompt + history_text)
                message_placeholder.markdown(response.text)
                
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            if st.session_state['user_role'] != "Guest":
                supabase.table("user_chat_history").insert({"student_id": st.session_state['username'], "role": "assistant", "content": response.text}).execute()

elif page == "📜 My History":
    st.title("📜 My Prediction History")
    st.markdown("Review your past stress assessments safely stored in the cloud.")
    
    try:
        # FETCH HISTORY FROM CLOUD DB
        res = supabase.table("user_history").select("created_at, Study_Hours_Per_Day, Sleep_Hours_Per_Day, GPA, Stress_Level").eq("student_id", st.session_state['username']).order("created_at", desc=True).execute()
        
        if not res.data:
            st.info("You don't have any saved records yet. Go to the 'AI Predictor' tab to take your first assessment!")
        else:
            history_df = pd.DataFrame(res.data)
            history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            history_df.rename(columns={
                'created_at': 'Date & Time',
                'Study_Hours_Per_Day': 'Study (Hours)',
                'Sleep_Hours_Per_Day': 'Sleep (Hours)',
                'Stress_Level': 'Predicted Stress Level'
            }, inplace=True)
            
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
    except Exception as e:
        st.error(f"Could not load history from cloud. Error: {e}")

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
                data = {"student_id": st.session_state['username'], "Study_Hours_Per_Day": s_study, "Sleep_Hours_Per_Day": s_sleep, "Social_Hours_Per_Day": s_social, "Physical_Activity_Hours_Per_Day": s_phys, "Extracurricular_Hours_Per_Day": s_extra, "GPA": s_gpa, "Stress_Level": s_stress}
                try:
                    supabase.table("user_history").insert(data).execute()
                    supabase.table("user_feedback").insert(data).execute()
                    st.success("Survey data added to your cloud history and training database!")
                except Exception as e:
                    st.error(f"Error saving to cloud: {e}")

elif page == "⚙️ Account Settings":
    st.title("⚙️ Account Settings")
    st.write("Manage your security credentials below.")
    
    with st.form("change_password_form"):
        st.subheader("Change Password")
        old_pwd = st.text_input("Current Password", type="password")
        new_pwd = st.text_input("New Password", type="password")
        confirm_pwd = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Update Password"):
            if new_pwd != confirm_pwd:
                st.error("New passwords do not match!")
            elif old_pwd and new_pwd:
                res = supabase.table("users").select("password_hash").eq("student_id", st.session_state['username']).execute()
                if res.data and check_hashes(old_pwd, res.data[0]['password_hash']):
                    supabase.table("users").update({"password_hash": make_hashes(new_pwd)}).eq("student_id", st.session_state['username']).execute()
                    st.success("Password updated successfully!")
                else:
                    st.error("Incorrect current password.")
            else:
                st.error("Please fill in all fields.")

elif page == "📈 Data Analysis":
    st.title("📈 Model Transparency")
    t1, t2, t3 = st.tabs(["Dataset", "Feature Importance", "Performance"])
    with t1:
        st.info("Because the data is now fully integrated into the cloud, this preview relies on live model training arrays.")
        try:
            conn = sqlite3.connect('student_stress.db', check_same_thread=False)
            df_display = pd.read_sql("SELECT * FROM training_data LIMIT 100", conn)
            fig = px.box(df_display, x="Stress_Level", y="Sleep_Hours_Per_Day", color="Stress_Level")
            st.plotly_chart(fig)
            conn.close()
        except:
            st.write("No local preview data found.")
    with t2:
        if model:
            imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
            st.plotly_chart(px.bar(imp, x='Importance', y='Feature', orientation='h'))
    with t3:
        if test_acc: st.metric("Model Accuracy", f"{test_acc*100:.2f}%")
        if model_cm is not None: st.plotly_chart(px.imshow(model_cm, text_auto=True, x=le.classes_, y=le.classes_, color_continuous_scale='Blues'))

elif page == "📊 Dashboard":
    st.title("Admin Dashboard & System Testing")
    try: 
        res = supabase.table("user_feedback").select("id", count="exact").execute()
        count_feed = res.count if res.count else 0
    except: 
        count_feed = 0
        
    col1, col2 = st.columns(2)
    col1.metric("Testing Accuracy", f"{test_acc*100:.1f}%")
    col2.metric("New Cloud Verified Feedback", count_feed)
    st.markdown("---")
    st.subheader("⚙️ Model Maintenance")
    if st.button("🔄 Retrain Model from Cloud Data"):
        if count_feed == 0: st.warning("No new data.")
        else:
            with st.spinner("Fetching data from cloud and retraining..."):
                st.cache_resource.clear()
                model, le, tr_acc, te_acc, cm, feats = train_internal_model(force_retrain=True)
                st.success("Retrained Successfully on Verified Cloud Feedback!")
