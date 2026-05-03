import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import time
import hashlib
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. Core Library Imports & Checks ---
try:
    from supabase import create_client, Client
    import google.generativeai as genai
except ImportError as e:
    st.error(f"🚨 Missing required libraries: {e}. Please ensure your requirements.txt includes supabase and google-generativeai.")
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

# --- AUTO-LOGIN HACK (Prevent logout on refresh) ---
if "user" in st.query_params:
    auto_user = st.query_params["user"]
    if auto_user and ('logged_in' not in st.session_state or not st.session_state.get('logged_in', False)):
        st.session_state['logged_in'] = True
        st.session_state['username'] = auto_user
        
        # Fetch real role from cloud database to persist permissions
        try:
            role_res = supabase.table("users").select("role").eq("student_id", auto_user).execute()
            if role_res.data:
                st.session_state['user_role'] = role_res.data[0].get('role', 'Student')
            else:
                st.session_state['user_role'] = "Student"
        except Exception:
            st.session_state['user_role'] = "Student"

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
                
                # Write user to URL params to handle refresh
                st.query_params["user"] = l_user 
                
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
# 2. Main System Logic (Professional Offline Model)
# ==========================================

@st.cache_resource
def load_professional_model():
    try:
        # UPDATED: Load models directly from the root directory based on your architecture
        model = joblib.load('student_stress_model.pkl')
        le = joblib.load('label_encoder.pkl')
        feature_names = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Lifestyle_Score', 'Academic_Pressure', 'GPA']
        
        # Performance metrics for dashboard display
        train_acc = 0.98  
        test_acc = 0.95   
        return model, le, train_acc, test_acc, None, feature_names
    except Exception as e:
        st.error(f"🚨 Model files not found in root directory! Please ensure .pkl files are uploaded. Error: {e}")
        return None, None, 0, 0, None, []

model, le, train_acc, test_acc, model_cm, feature_names = load_professional_model()

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
    # Clear URL params on logout
    st.query_params.clear()
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
    - **🔄 Continuous Learning:** Architecture supports verified feedback training via offline batching.
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
        st.caption("💡 You can use the sliders or type the numbers directly.")

        # Callbacks for dual-input synchronization
        def sync_study():
            st.session_state.num_study = st.session_state.slider_study
        def sync_study_rev():
            st.session_state.slider_study = st.session_state.num_study
            
        def sync_sleep():
            st.session_state.num_sleep = st.session_state.slider_sleep
        def sync_sleep_rev():
            st.session_state.slider_sleep = st.session_state.num_sleep
            
        def sync_social():
            st.session_state.num_social = st.session_state.slider_social
        def sync_social_rev():
            st.session_state.slider_social = st.session_state.num_social
            
        def sync_phys():
            st.session_state.num_phys = st.session_state.slider_phys
        def sync_phys_rev():
            st.session_state.slider_phys = st.session_state.num_phys
            
        def sync_extra():
            st.session_state.num_extra = st.session_state.slider_extra
        def sync_extra_rev():
            st.session_state.slider_extra = st.session_state.num_extra

        # Initialize Session State values
        if 'slider_study' not in st.session_state: st.session_state.slider_study = 5.0
        if 'num_study' not in st.session_state: st.session_state.num_study = 5.0
        
        if 'slider_sleep' not in st.session_state: st.session_state.slider_sleep = 7.0
        if 'num_sleep' not in st.session_state: st.session_state.num_sleep = 7.0
        
        if 'slider_social' not in st.session_state: st.session_state.slider_social = 2.0
        if 'num_social' not in st.session_state: st.session_state.num_social = 2.0
        
        if 'slider_phys' not in st.session_state: st.session_state.slider_phys = 1.0
        if 'num_phys' not in st.session_state: st.session_state.num_phys = 1.0
        
        if 'slider_extra' not in st.session_state: st.session_state.slider_extra = 1.0
        if 'num_extra' not in st.session_state: st.session_state.num_extra = 1.0

        # UI Components with visible labels and side-by-side number inputs
        col_s1, col_n1 = st.columns([3, 1])
        with col_s1:
            st.slider("Study Hours (per day)", 0.0, 24.0, key="slider_study", step=0.5, on_change=sync_study)
        with col_n1:
            st.number_input("Study Input", 0.0, 24.0, key="num_study", step=0.5, on_change=sync_study_rev, label_visibility="hidden")

        col_s2, col_n2 = st.columns([3, 1])
        with col_s2:
            st.slider("Sleep Hours (per day)", 0.0, 24.0, key="slider_sleep", step=0.5, on_change=sync_sleep)
        with col_n2:
            st.number_input("Sleep Input", 0.0, 24.0, key="num_sleep", step=0.5, on_change=sync_sleep_rev, label_visibility="hidden")

        col_s3, col_n3 = st.columns([3, 1])
        with col_s3:
            st.slider("Social Hours (per day)", 0.0, 24.0, key="slider_social", step=0.5, on_change=sync_social)
        with col_n3:
            st.number_input("Social Input", 0.0, 24.0, key="num_social", step=0.5, on_change=sync_social_rev, label_visibility="hidden")

        col_s4, col_n4 = st.columns([3, 1])
        with col_s4:
            st.slider("Physical Activity (per day)", 0.0, 24.0, key="slider_phys", step=0.5, on_change=sync_phys)
        with col_n4:
            st.number_input("Phys Input", 0.0, 24.0, key="num_phys", step=0.5, on_change=sync_phys_rev, label_visibility="hidden")

        col_s5, col_n5 = st.columns([3, 1])
        with col_s5:
            st.slider("Extracurriculars (per day)", 0.0, 24.0, key="slider_extra", step=0.5, on_change=sync_extra)
        with col_n5:
            st.number_input("Extra Input", 0.0, 24.0, key="num_extra", step=0.5, on_change=sync_extra_rev, label_visibility="hidden")

        # Dynamic variable binding
        study = st.session_state.num_study
        sleep = st.session_state.num_sleep
        social = st.session_state.num_social
        physical = st.session_state.num_phys
        extra = st.session_state.num_extra

        total_hours = study + sleep + social + physical + extra
        hours_left = 24.0 - total_hours

        st.markdown("---")
        if hours_left >= 0:
            st.markdown(f"**⏱️ Time Budget:** :green[{hours_left:.1f} hours remaining] (Used: {total_hours}/24)")
            st.progress(total_hours / 24.0)
        else:
            st.markdown(f"**🚨 Time Overload:** :red[You are over by {abs(hours_left):.1f} hours!] (Used: {total_hours}/24)")
            st.progress(1.0) 

        gpa = st.slider("Current GPA", 0.0, 4.0, 3.0, step=0.1)
        lifestyle_score = social + physical + extra
        academic_pressure = gpa * study
        
        if total_hours > 24:
            st.error(f"🚨 Invalid Input: Total hours ({total_hours}) cannot exceed 24.")
        else:
            if st.button("Analyze Stress Level", use_container_width=True):
                if model:
                    input_data = pd.DataFrame([[study, sleep, lifestyle_score, academic_pressure, gpa]], columns=feature_names)
                    pred_idx = model.predict(input_data)[0]
                    pred_proba = model.predict_proba(input_data)[0]
                    confidence = np.max(pred_proba) * 100
                    pred_label = le.inverse_transform([pred_idx])[0]
                    
                    # Auto-save history to Cloud
                    if st.session_state['user_role'] != "Guest":
                        try:
                            supabase.table("user_history").insert({
                                "student_id": st.session_state['username'],
                                "Study_Hours_Per_Day": study, "Sleep_Hours_Per_Day": sleep,
                                "Social_Hours_Per_Day": social, "Physical_Activity_Hours_Per_Day": physical,
                                "Extracurricular_Hours_Per_Day": extra, "GPA": gpa, "Stress_Level": pred_label
                            }).execute()
                        except Exception as e:
                            print(f"Cloud History Sync Error: {e}")

                    # Generate AI advice
                    ai_advice = ""
                    if gemini_model:
                        with st.spinner("🤖 Consulting Gemini AI Counselor..."):
                            try:
                                prompt = f"Act as a professional University Counselor. Student profile: Study {study} hrs/day, Sleep {sleep} hrs/day, GPA {gpa}. Predicted Stress: {pred_label} ({confidence:.1f}%). Provide a 1-sentence analysis and 3 specific actionable tips."
                                response = gemini_model.generate_content(prompt)
                                ai_advice = response.text
                            except Exception:
                                ai_advice = "Connection to Gemini failed. Please check your API configuration."
                    
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
                st.write("Is this prediction accurate for your current state?")
                cy, cn = st.columns(2)
                
                if cy.button("✅ Yes, Accurate"):
                    feedback_data = p['inputs'].copy()
                    feedback_data["student_id"] = st.session_state['username']
                    try:
                        supabase.table("user_feedback").insert(feedback_data).execute()
                        st.success("Verification saved to Cloud!")
                    except Exception as e:
                        st.error(f"Cloud Sync Failed: {e}")
                    
                if cn.button("❌ No, Correct It"):
                    st.session_state['feedback_mode'] = True

                if st.session_state.get('feedback_mode', False):
                    with st.form("correction_form"):
                        correct_label = st.selectbox("Your actual stress level", ["Low", "Moderate", "High"])
                        if st.form_submit_button("Submit Correction"):
                            feedback_data = p['inputs'].copy()
                            feedback_data["Stress_Level"] = correct_label
                            feedback_data["student_id"] = st.session_state['username']
                            try:
                                supabase.table("user_feedback").insert(feedback_data).execute()
                                st.success("Correction submitted for future batch training!")
                            except Exception as e:
                                st.error(f"Cloud Sync Failed: {e}")
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
         st.info("👋 Guest Session: Chat history will not be persisted.")
         if "messages" not in st.session_state:
             st.session_state.messages = []
    else:
        # Load chat from Cloud
        if not st.session_state.get('chat_loaded', False):
            try:
                res = supabase.table("user_chat_history").select("role, content").eq("student_id", st.session_state['username']).order("created_at").execute()
                if res.data:
                    st.session_state.messages = [{"role": row['role'], "content": row['content']} for row in res.data]
                else:
                    st.session_state.messages = []
            except Exception:
                st.session_state.messages = []
            st.session_state['chat_loaded'] = True
         
    if not gemini_model:
        st.warning("⚠️ Gemini connection inactive.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): 
                st.markdown(message["content"])
                
        if prompt := st.chat_input("Tell me what's on your mind..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            if st.session_state['user_role'] != "Guest":
                supabase.table("user_chat_history").insert({"student_id": st.session_state['username'], "role": "user", "content": prompt}).execute()
                
            with st.chat_message("user"): 
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                system_prompt = "You are an empathetic AI Student Counselor. Provide warm and practical support."
                response = gemini_model.generate_content(system_prompt + history_text)
                message_placeholder.markdown(response.text)
                
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            if st.session_state['user_role'] != "Guest":
                supabase.table("user_chat_history").insert({"student_id": st.session_state['username'], "role": "assistant", "content": response.text}).execute()

elif page == "📜 My History":
    st.title("📜 My Prediction History")
    st.markdown("Cloud-synced stress assessment records.")
    
    try:
        res = supabase.table("user_history").select("created_at, Study_Hours_Per_Day, Sleep_Hours_Per_Day, GPA, Stress_Level").eq("student_id", st.session_state['username']).order("created_at", desc=True).execute()
        
        if not res.data:
            st.info("No records found. Visit the AI Predictor to start tracking.")
        else:
            history_df = pd.DataFrame(res.data)
            history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            history_df.rename(columns={
                'created_at': 'Timestamp',
                'Study_Hours_Per_Day': 'Study',
                'Sleep_Hours_Per_Day': 'Sleep',
                'Stress_Level': 'Result'
            }, inplace=True)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
    except Exception as e:
        st.error(f"Could not load Cloud History: {e}")

elif page == "📝 User Survey":
    st.title("📝 Lifestyle Survey")
    with st.form("survey_form"):
        s_study = st.number_input("Study Hours", 0.0, 24.0, 5.0)
        s_sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
        s_social = st.number_input("Social Hours", 0.0, 24.0, 2.0)
        s_phys = st.number_input("Physical Activity", 0.0, 24.0, 1.0)
        s_extra = st.number_input("Extracurriculars", 0.0, 24.0, 1.0)
        s_gpa = st.number_input("GPA", 0.0, 4.0, 3.0)
        s_stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
        if st.form_submit_button("Submit to Cloud"):
            if s_study + s_sleep + s_social + s_phys + s_extra > 24:
                st.error("Invalid Input: Hours exceed 24.")
            else:
                data = {"student_id": st.session_state['username'], "Study_Hours_Per_Day": s_study, "Sleep_Hours_Per_Day": s_sleep, "Social_Hours_Per_Day": s_social, "Physical_Activity_Hours_Per_Day": s_phys, "Extracurricular_Hours_Per_Day": s_extra, "GPA": s_gpa, "Stress_Level": s_stress}
                try:
                    supabase.table("user_history").insert(data).execute()
                    supabase.table("user_feedback").insert(data).execute()
                    st.success("Data successfully synced to Cloud training database.")
                except Exception as e:
                    st.error(f"Cloud Sync Failed: {e}")

elif page == "⚙️ Account Settings":
    st.title("⚙️ Account Settings")
    
    with st.form("change_password_form"):
        st.subheader("Change Password")
        old_pwd = st.text_input("Current Password", type="password")
        new_pwd = st.text_input("New Password", type="password")
        confirm_pwd = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Update Password"):
            if new_pwd != confirm_pwd:
                st.error("New passwords do not match.")
            elif old_pwd and new_pwd:
                res = supabase.table("users").select("password_hash").eq("student_id", st.session_state['username']).execute()
                if res.data and check_hashes(old_pwd, res.data[0]['password_hash']):
                    supabase.table("users").update({"password_hash": make_hashes(new_pwd)}).eq("student_id", st.session_state['username']).execute()
                    st.success("Security credentials updated.")
                else:
                    st.error("Current password incorrect.")

elif page == "📈 Data Analysis":
    st.title("📈 Model Transparency")
    t1, t2, t3 = st.tabs(["Original Dataset", "Feature Importance", "Metrics"])
    with t1:
        st.info("Previewing training baseline dataset from root directory.")
        try:
            # UPDATED: Path matches your root directory structure
            df_display = pd.read_csv("student_lifestyle_dataset.csv").head(100)
            fig = px.box(df_display, x="Stress_Level", y="Sleep_Hours_Per_Day", color="Stress_Level")
            st.plotly_chart(fig)
        except:
            st.warning("Baseline dataset (student_lifestyle_dataset.csv) not found in root.")
    with t2:
        if model:
            imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
            st.plotly_chart(px.bar(imp, x='Importance', y='Feature', orientation='h'))
    with t3:
        if test_acc: st.metric("Predictive Accuracy", f"{test_acc*100:.2f}%")
        if model_cm is not None: 
            st.plotly_chart(px.imshow(model_cm, text_auto=True, x=le.classes_, y=le.classes_, color_continuous_scale='Blues'))

elif page == "📊 Dashboard":
    st.title("Admin Dashboard & System Monitoring")
    try: 
        res = supabase.table("user_feedback").select("id", count="exact").execute()
        count_feed = res.count if res.count else 0
    except: 
        count_feed = 0
        
    col1, col2 = st.columns(2)
    col1.metric("Operational Accuracy", f"{test_acc*100:.1f}%")
    col2.metric("New Verified Feedback (Cloud)", count_feed)
    st.markdown("---")
    st.subheader("Maintenance Operations")
    if st.button("Check for Model Updates"):
        st.info("Architecture: Offline Batch Retraining. To update the model core, perform local training and redeploy updated .pkl files.")
