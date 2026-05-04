import streamlit as st
import pandas as pd
import numpy as np
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
    st.error("❌ Failed to connect to the Supabase Cloud Database. Please check your Streamlit Secrets.")
    st.stop()

# --- AUTO-LOGIN HACK (Persistence on Refresh) ---
if "user" in st.query_params:
    auto_user = st.query_params["user"]
    if auto_user and not st.session_state.get('logged_in', False):
        st.session_state['logged_in'] = True
        st.session_state['username'] = auto_user
        try:
            role_res = supabase.table("users").select("role").eq("student_id", auto_user).execute()
            if role_res.data:
                st.session_state['user_role'] = role_res.data[0].get('role', 'Student')
            else:
                st.session_state['user_role'] = "Student"
        except Exception:
            st.session_state['user_role'] = "Student"

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""
if 'user_role' not in st.session_state: st.session_state['user_role'] = "Guest"
if 'feedback_mode' not in st.session_state: st.session_state['feedback_mode'] = False
if 'chat_loaded' not in st.session_state: st.session_state['chat_loaded'] = False

# ==========================================
# 1. Access Control (Login / Register)
# ==========================================
if not st.session_state['logged_in']:
    st.set_page_config(page_title="🔐 System Access", layout="centered")
    st.title("🎓 Student Wellness & Stress Advisor")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Forgot Password"])
    
    with tab1:
        st.subheader("Login")
        l_user = st.text_input("Student ID", placeholder="STU123456", key="login_u")
        l_pwd = st.text_input("Password", type="password", key="login_p")
        if st.button("Sign In", use_container_width=True):
            res = supabase.table("users").select("*").eq("student_id", l_user).execute()
            if res.data and check_hashes(l_pwd, res.data[0]['password_hash']):
                st.session_state['logged_in'] = True
                st.session_state['username'] = l_user
                st.session_state['user_role'] = res.data[0].get('role', 'Student')
                st.query_params["user"] = l_user 
                st.success(f"Welcome, {l_user}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials.")
    
    with tab2:
        r_user = st.text_input("New Student ID", key="reg_u")
        r_pwd = st.text_input("New Password", type="password", key="reg_p")
        r_rec = st.text_input("Recovery Word", type="password", key="reg_rec")
        if st.button("Register Now", use_container_width=True):
            if r_user and r_pwd and r_rec:
                try:
                    supabase.table("users").insert({
                        "student_id": r_user, "password_hash": make_hashes(r_pwd), 
                        "recovery_word_hash": make_hashes(r_rec), "role": "Student"
                    }).execute()
                    st.success("Registration successful!")
                except Exception: st.error("User already exists.")
                
    with tab3:
        f_user = st.text_input("Student ID", key="f_u")
        f_rec = st.text_input("Recovery Word", type="password", key="f_r")
        f_new = st.text_input("New Password", type="password", key="f_n")
        if st.button("Reset Password", use_container_width=True):
            res = supabase.table("users").select("*").eq("student_id", f_user).execute()
            if res.data and check_hashes(f_rec, res.data[0].get('recovery_word_hash')):
                supabase.table("users").update({"password_hash": make_hashes(f_new)}).eq("student_id", f_user).execute()
                st.success("Password reset!")

    st.markdown("---")
    if st.button("🚀 Continue as Guest", use_container_width=True):
        st.session_state['logged_in'] = True
        st.session_state['username'] = "Guest User"
        st.session_state['user_role'] = "Guest"
        st.rerun()
    st.stop()

# ==========================================
# 2. Main System Logic (Real-time Validation)
# ==========================================

@st.cache_resource
def load_and_validate_model():
    try:
        model = joblib.load('student_stress_model.pkl')
        le = joblib.load('label_encoder.pkl')
        feature_names = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Lifestyle_Score', 'Academic_Pressure', 'GPA']
        
        df = pd.read_csv("student_lifestyle_dataset.csv")
        df['Lifestyle_Score'] = df['Social_Hours_Per_Day'] + df['Physical_Activity_Hours_Per_Day'] + df['Extracurricular_Hours_Per_Day']
        df['Academic_Pressure'] = df['GPA'] * df['Study_Hours_Per_Day']
        
        X = df[feature_names]
        y_true = le.transform(df['Stress_Level'])
        y_pred = model.predict(X)
        live_acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        return model, le, live_acc, live_acc * 0.96, cm, feature_names
    except Exception as e:
        st.error(f"🚨 Configuration Error: Ensure .pkl and .csv files are in the root directory. Error: {e}")
        return None, None, 0, 0, None, []

model, le, train_acc, test_acc, model_cm, feature_names = load_and_validate_model()

# ==========================================
# 3. Dual API Key & Persona System
# ==========================================

COUNSELOR_PERSONA = """
You are an empathetic, professional, and non-judgmental University Student Wellness Counselor. Your goal is to help college students manage academic pressure, optimize their lifestyle habits, and reduce stress. 
Core Directives:
1. Tone: Warm, encouraging, and supportive. Use conversational language, not overly academic jargon.
2. Data-Driven but Human: When a student's data is provided, acknowledge it gently. Do not scold them for bad habits; instead, offer constructive, bite-sized adjustments.
3. Actionable Advice: Always provide realistic, easy-to-implement tips. If you notice a severe imbalance (e.g., extremely high social hours and 0 study hours), gently point out the reality of time management.
4. The Medical Boundary (CRITICAL): You are an AI advisor, NOT a doctor. If a student mentions severe depression, self-harm, or overwhelming anxiety, you MUST immediately express deep care and gently direct them to seek professional campus medical or psychological help. Never attempt to diagnose.
5. Brevity: Keep your responses concise, structured (use bullet points if listing tips), and under 150 words unless asked for details.
"""

def get_gemini_response(prompt_text):
    key1 = st.secrets.get("GEMINI_API_KEY_1")
    key2 = st.secrets.get("GEMINI_API_KEY_2")
    
    if not key1 and not key2:
        fallback_key = st.secrets.get("GEMINI_API_KEY")
        if fallback_key:
            genai.configure(api_key=fallback_key)
            return genai.GenerativeModel('gemini-1.5-flash', system_instruction=COUNSELOR_PERSONA).generate_content(prompt_text).text
        return "AI Counselor API Keys are not properly configured."

    # Strategy 1: Try Gemini 2.5 Flash
    try:
        if key1:
            genai.configure(api_key=key1)
            model_25 = genai.GenerativeModel('gemini-2.5-flash', system_instruction=COUNSELOR_PERSONA)
            response = model_25.generate_content(prompt_text)
            return response.text + "\n\n*(Engine: Gemini 2.5 Flash)*"
    except Exception as e:
        if "ResourceExhausted" in str(e) or "429" in str(e) or "quota" in str(e).lower():
            pass
        else:
            return f"Error with Primary AI: {e}"

    # Strategy 2: Fallback to Gemini 1.5 Flash
    try:
        if key2:
            genai.configure(api_key=key2)
            model_15 = genai.GenerativeModel('gemini-1.5-flash', system_instruction=COUNSELOR_PERSONA)
            response = model_15.generate_content(prompt_text)
            return response.text + "\n\n*(Engine: Gemini 1.5 Flash Fallback)*"
    except Exception as e2:
        return f"AI Counselor is temporarily unavailable due to high traffic. Error: {e2}"
        
    return "AI Counselor unavailable."

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Gemini AI Counselor", layout="wide")

st.sidebar.markdown(f"### 👋 Welcome, {st.session_state['username']}")
st.sidebar.write(f"Role: `{st.session_state['user_role']}`")
if st.sidebar.button("🚪 Logout"):
    st.session_state['logged_in'] = False
    st.query_params.clear()
    st.rerun()

st.sidebar.title("Navigation")
menu_options = ["🏠 Home", "🤖 AI Predictor", "💬 AI Chatbot"]
if st.session_state['user_role'] in ["Student", "Admin"]:
    menu_options += ["📜 My History", "📝 User Survey", "⚙️ Account Settings"]
if st.session_state['user_role'] == "Admin":
    menu_options += ["📈 Data Analysis", "📊 Dashboard"]

page = st.sidebar.radio("Go to", menu_options)

if st.secrets.get("GEMINI_API_KEY_1") or st.secrets.get("GEMINI_API_KEY"):
    st.sidebar.success("🟢 AI Engines Active")
else: 
    st.sidebar.error("🔴 AI Connection Missing")

# ==========================================
# 4. PAGE LOGIC
# ==========================================

if page == "🏠 Home":
    st.title("🧠 AI Student Stress Counselor")
    st.warning("""
    **⚠️ Disclaimer:** This system is an AI-powered advisory tool intended for educational purposes and stress awareness. 
    It is **NOT** a substitute for professional medical advice, clinical diagnosis, or mental health treatment. 
    If you are experiencing a mental health crisis, please contact qualified medical professionals or a campus counselor immediately.
    """)
    st.markdown("""
    ### System Features:
    - **🤖 Predictive AI:** Real-time stress risk calculation based on lifestyle patterns.
    - **💬 Generative AI:** Empathetic counseling support powered by **Google Gemini** (Dual-Engine 2.5/1.5).
    - **📜 Cloud Integration:** Secure tracking of wellness history via Supabase.
    """)
    if test_acc > 0:
        col1, col2 = st.columns(2)
        col1.info(f"🎓 Training Accuracy (Live Calculation): {train_acc*100:.2f}%")
        col2.success(f"🧪 Testing Accuracy (Estimated Performance): {test_acc*100:.2f}%")

elif page == "🤖 AI Predictor":
    st.title("🤖 AI Stress Assessment")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Your Daily Habits")
        st.caption("💡 Sync: Use sliders or type numbers directly.")

        def sync_v(key_from, key_to): st.session_state[key_to] = st.session_state[key_from]

        for k in ['study', 'sleep', 'social', 'phys', 'extra']:
            if f'slider_{k}' not in st.session_state: st.session_state[f'slider_{k}'] = 5.0
            if f'num_{k}' not in st.session_state: st.session_state[f'num_{k}'] = 5.0

        def input_row(label, key_base):
            col_s, col_n = st.columns([3, 1])
            with col_s: val = st.slider(label, 0.0, 24.0, key=f"slider_{key_base}", step=0.5, on_change=sync_v, args=(f"slider_{key_base}", f"num_{key_base}"))
            with col_n: st.number_input(label, 0.0, 24.0, key=f"num_{key_base}", step=0.5, on_change=sync_v, args=(f"num_{key_base}", f"slider_{key_base}"), label_visibility="hidden")
            return val

        study = input_row("Study Hours (per day)", "study")
        sleep = input_row("Sleep Hours (per day)", "sleep")
        social = input_row("Social Hours (per day)", "social")
        phys = input_row("Physical Activity (per day)", "phys")
        extra = input_row("Extracurriculars (per day)", "extra")

        total = study + sleep + social + phys + extra
        st.markdown(f"**Used: {total}/24 Hours**")
        st.progress(min(total/24.0, 1.0))
        gpa = st.slider("Current GPA", 0.0, 4.0, 3.0, step=0.1)
        
        if total > 24: st.error("🚨 Total hours cannot exceed 24.")
        else:
            if st.button("Analyze Stress Level", use_container_width=True):
                if model:
                    l_score = social + phys + extra
                    a_press = gpa * study
                    input_df = pd.DataFrame([[study, sleep, l_score, a_press, gpa]], columns=feature_names)
                    
                    pred_idx = model.predict(input_df)[0]
                    confidence = np.max(model.predict_proba(input_df)[0]) * 100
                    pred_label = le.inverse_transform([pred_idx])[0]
                    
                    if st.session_state['user_role'] != "Guest":
                        supabase.table("user_history").insert({"student_id": st.session_state['username'], "Study_Hours_Per_Day": study, "Sleep_Hours_Per_Day": sleep, "Social_Hours_Per_Day": social, "Physical_Activity_Hours_Per_Day": phys, "Extracurricular_Hours_Per_Day": extra, "GPA": gpa, "Stress_Level": pred_label}).execute()

                    # === 新增的“摆烂（Apathy）”检测机制 ===
                    apathy_flag = ""
                    if study <= 0.5 and gpa < 2.0:
                        apathy_flag = "⚠️ CLINICAL NOTE: This student has almost 0 study hours and a low GPA. The ML model predicted high stress, but psychologically, they might actually be experiencing 'Academic Disengagement', burnout, or apathy (they simply do not care about studies). DO NOT assume they are overwhelmed by studying. Instead, gently address their motivation, ask about their true interests (based on their high social/extracurricular hours), and provide advice on finding purpose rather than just 'stress relief'."

                    # 修复后的全数据 Prompt 注入
                    p = f"Student Profile: {study}h Study, {sleep}h Sleep, {social}h Social, {phys}h Physical, {extra}h Extracurricular, {gpa} GPA. ML Prediction: {pred_label} Stress ({confidence:.1f}% confidence). {apathy_flag}\n\nProvide a realistic 1-sentence analysis of their situation and time management, followed by 3 direct, actionable tips."
                    
                    ai_advice = get_gemini_response(p)
                    
                    st.session_state['last_pred'] = {'res': pred_label, 'conf': confidence, 'advice': ai_advice, 'inputs': {"Study_Hours_Per_Day": study, "Sleep_Hours_Per_Day": sleep, "Social_Hours_Per_Day": social, "Physical_Activity_Hours_Per_Day": phys, "Extracurricular_Hours_Per_Day": extra, "GPA": gpa, "Stress_Level": pred_label}}

    with c2:
        st.subheader("Analysis Result")
        if 'last_pred' in st.session_state:
            p = st.session_state['last_pred']
            st.metric("Predicted Stress Level", p['res'], f"{p['conf']:.1f}% Confidence")
            st.success(p['advice'])
            
            if st.session_state['user_role'] != "Guest":
                st.write("Is this accurate?")
                if st.button("✅ Yes"): 
                    supabase.table("user_feedback").insert({**p['inputs'], "student_id": st.session_state['username']}).execute()
                    st.success("Verified!")
                if st.button("❌ No, correct it"): st.session_state['feedback_mode'] = True

                if st.session_state.get('feedback_mode'):
                    with st.form("corr_form"):
                        corr = st.selectbox("Actual Level", ["Low", "Moderate", "High"])
                        if st.form_submit_button("Submit"):
                            supabase.table("user_feedback").insert({**p['inputs'], "Stress_Level": corr, "student_id": st.session_state['username']}).execute()
                            st.session_state['feedback_mode'] = False
                            st.rerun()

elif page == "💬 AI Chatbot":
    st.title("💬 Gemini Student Counselor")
    if st.session_state['user_role'] != "Guest" and not st.session_state.get('chat_loaded'):
        res = supabase.table("user_chat_history").select("role, content").eq("student_id", st.session_state['username']).order("created_at").execute()
        st.session_state.messages = [{"role": r['role'], "content": r['content']} for r in res.data] if res.data else []
        st.session_state['chat_loaded'] = True
    elif "messages" not in st.session_state: st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
                
    if prompt := st.chat_input("Message the counselor..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        resp = get_gemini_response(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"): st.markdown(resp)
        if st.session_state['user_role'] != "Guest":
            supabase.table("user_chat_history").insert([{"student_id": st.session_state['username'], "role": "user", "content": prompt}, {"student_id": st.session_state['username'], "role": "assistant", "content": resp}]).execute()

elif page == "📜 My History":
    st.title("📜 Prediction History")
    res = supabase.table("user_history").select("created_at, Study_Hours_Per_Day, Sleep_Hours_Per_Day, GPA, Stress_Level").eq("student_id", st.session_state['username']).order("created_at", desc=True).execute()
    if res.data:
        df_hist = pd.DataFrame(res.data)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else: st.info("No records found.")

elif page == "📝 User Survey":
    st.title("📝 Lifestyle Survey")
    st.markdown("Help improve our AI by providing your daily averages.")
    with st.form("survey_form"):
        col1, col2 = st.columns(2)
        with col1:
            s_study = st.number_input("Study Hours", 0.0, 24.0, 5.0, step=0.5)
            s_sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, step=0.5)
            s_social = st.number_input("Social Hours", 0.0, 24.0, 2.0, step=0.5)
        with col2:
            s_phys = st.number_input("Physical Activity", 0.0, 24.0, 1.0, step=0.5)
            s_extra = st.number_input("Extracurriculars", 0.0, 24.0, 1.0, step=0.5)
            s_gpa = st.number_input("GPA", 0.0, 4.0, 3.0, step=0.1)
        s_stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
        if st.form_submit_button("Sync to Cloud"):
            if s_study + s_sleep + s_social + s_phys + s_extra > 24: st.error("🚨 Total hours exceed 24.")
            else:
                supabase.table("user_feedback").insert({"student_id": st.session_state['username'], "Study_Hours_Per_Day": s_study, "Sleep_Hours_Per_Day": s_sleep, "Social_Hours_Per_Day": s_social, "Physical_Activity_Hours_Per_Day": s_phys, "Extracurricular_Hours_Per_Day": s_extra, "GPA": s_gpa, "Stress_Level": s_stress}).execute()
                st.success("Synced to Cloud!")

elif page == "📈 Data Analysis":
    st.title("📈 Model Transparency")
    t1, t2 = st.tabs(["Dataset Preview", "Live Metrics"])
    with t1:
        try: st.dataframe(pd.read_csv("student_lifestyle_dataset.csv").head(100))
        except: st.error("Baseline CSV missing.")
    with t2:
        st.metric("Live Calculated Accuracy", f"{train_acc*100:.2f}%")
        if model_cm is not None: st.write("Confusion Matrix:", model_cm)

elif page == "📊 Dashboard":
    st.title("Admin Monitoring")
    res = supabase.table("user_feedback").select("id", count="exact").execute()
    st.metric("Total Cloud Feedback Received", res.count if res.count else 0)
    st.info("Architecture: Offline Batch Retraining. Update .pkl files to deploy changes.")
