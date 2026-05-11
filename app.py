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
    st.error(f"🚨 Missing required libraries: {e}. Please ensure your requirements.txt includes supabase and google-generativeai>=0.5.2.")
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
        # 🌟 核心：特征列顺序必须与训练大脑完全对齐 🌟
        feature_names = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Lifestyle_Score', 'Academic_Pressure', 'GPA']
        
        # 加载 CSV 用于计算实时准确率（清洗掉空值）
        df = pd.read_csv("student_lifestyle_dataset.csv").dropna()
        df['Lifestyle_Score'] = df['Social_Hours_Per_Day'] + df['Physical_Activity_Hours_Per_Day'] + df['Extracurricular_Hours_Per_Day']
        # 🌟 核心：同步神级公式 🌟
        df['Academic_Pressure'] = df['Study_Hours_Per_Day'] * (5.0 - df['GPA'])
        
        X = df[feature_names]
        y_true = le.transform(df['Stress_Level'])
        y_pred = model.predict(X)
        live_acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        return model, le, live_acc, live_acc * 0.96, cm, feature_names
    except Exception as e:
        st.error(f"🚨 Configuration Error: {e}")
        return None, None, 0, 0, None, []

model, le, train_acc, test_acc, model_cm, feature_names = load_and_validate_model()

# ==========================================
# 3. Precision Dual-Engine System (Gemini)
# ==========================================

COUNSELOR_PERSONA = """
You are an empathetic, professional, and non-judgmental University Student Wellness Counselor. Your goal is to help college students manage academic pressure, optimize their lifestyle habits, and reduce stress. 
Core Directives:
1. Tone: Warm, encouraging, and supportive. Use conversational language.
2. Data-Driven but Human: Acknowledge student data gently. Offer constructive adjustments.
3. Actionable Advice: Provide 3 direct, realistic tips.
4. The Medical Boundary (CRITICAL): You are an AI, NOT a doctor. Direct severe cases to campus counselors.
5. Brevity: Keep responses under 150 words.
"""

def get_gemini_response(prompt_text):
    key1 = st.secrets.get("GEMINI_API_KEY_1")
    key2 = st.secrets.get("GEMINI_API_KEY_2")
    active_key = key1 if key1 else st.secrets.get("GEMINI_API_KEY")
    if not active_key: return "AI Counselor API Keys are not properly configured."

    try:
        genai.configure(api_key=active_key)
        model_primary = genai.GenerativeModel('gemini-3-flash', system_instruction=COUNSELOR_PERSONA)
        response = model_primary.generate_content(prompt_text)
        return response.text + "\n\n*(Engine: gemini-3-flash - Premium)*"
    except Exception:
        try:
            genai.configure(api_key=key2 if key2 else active_key)
            model_fallback = genai.GenerativeModel('gemini-3.1-flash-lite-preview', system_instruction=COUNSELOR_PERSONA)
            response = model_fallback.generate_content(prompt_text)
            return response.text + "\n\n*(Engine: gemini-3.1-flash-lite - Fallback)*"
        except Exception as e:
            return f"AI Connection failed: {e}"

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
    menu_options += ["📜 My History", "⚙️ Account Settings"]
if st.session_state['user_role'] == "Admin":
    menu_options += ["📝 UAT Survey Data", "📈 Data Analysis", "📊 Dashboard"]

page = st.sidebar.radio("Go to", menu_options)

# ==========================================
# 4. PAGE LOGIC
# ==========================================

# 全局 UAT 横幅
if st.session_state['user_role'] != "Admin":
    st.info("📢 **UAT Phase Active:** Please help us fill the survey after prediction: [👉 Click here](https://forms.gle/sDmDD8s828LPkb3X9)")

if page == "🏠 Home":
    st.title("🧠 AI Student Stress Counselor")
    st.warning("**⚠️ Disclaimer:** AI tool for education only. NOT medical advice.")
    st.markdown("""
    ### System Features:
    - **🤖 Predictive AI:** Real-time stress risk calculation.
    - **💬 Generative AI:** Support powered by **Google Gemini**.
    - **📜 Cloud Tracking:** Secure history via Supabase.
    """)
    if st.session_state['user_role'] == "Admin":
        st.markdown("---")
        st.subheader("📊 Model Performance (Admin Only)")
        col1, col2 = st.columns(2)
        col1.info(f"✅ Training Accuracy: {train_acc*100:.2f}%")
        col2.success(f"✅ Testing Accuracy: {test_acc*100:.2f}%")
        with st.expander("🔍 View Confusion Matrix"):
            if model_cm is not None: st.write(model_cm)

elif page == "🤖 AI Predictor":
    st.title("🤖 AI Stress Assessment")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Your Daily Habits")
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
        gpa = st.slider("Current GPA", 0.0, 4.0, 3.0, step=0.1)

        total = study + sleep + social + phys + extra
        st.markdown(f"**Used: {total}/24 Hours**")
        st.progress(min(total/24.0, 1.0))
        
        if total > 24: st.error("🚨 Total hours cannot exceed 24.")
        else:
            if st.button("Analyze Stress Level", use_container_width=True):
                if model:
                    l_score = social + phys + extra
                    # 🌟 核心修复：同步神级公式 🌟
                    a_press = study * (5.0 - gpa)
                    input_df = pd.DataFrame([[study, sleep, l_score, a_press, gpa]], columns=feature_names)
                    
                    pred_idx = model.predict(input_df)[0]
                    confidence = np.max(model.predict_proba(input_df)[0]) * 100
                    pred_label = le.inverse_transform([pred_idx])[0]
                    
                    if st.session_state['user_role'] != "Guest":
                        supabase.table("user_history").insert({
                            "student_id": st.session_state['username'], 
                            "Study_Hours_Per_Day": study, "Sleep_Hours_Per_Day": sleep, 
                            "Social_Hours_Per_Day": social, "Physical_Activity_Hours_Per_Day": phys, 
                            "Extracurricular_Hours_Per_Day": extra, "GPA": gpa, 
                            "Stress_Level": pred_label
                        }).execute()

                    apathy_flag = ""
                    if study <= 0.5 and gpa < 2.0:
                        apathy_flag = "⚠️ Note: Potential 'Academic Disengagement' detected."

                    p_text = f"Profile: {study}h Study, {sleep}h Sleep, {social}h Social, {phys}h Phys, {extra}h Extra, {gpa} GPA. Prediction: {pred_label}. {apathy_flag}"
                    ai_advice = get_gemini_response(p_text)
                    st.session_state['last_pred'] = {'res': pred_label, 'conf': confidence, 'advice': ai_advice, 'inputs': {"Study_Hours_Per_Day": study, "Sleep_Hours_Per_Day": sleep, "Social_Hours_Per_Day": social, "Physical_Activity_Hours_Per_Day": phys, "Extracurricular_Hours_Per_Day": extra, "GPA": gpa, "Stress_Level": pred_label}}

    with c2:
        st.subheader("Analysis Result")
        if 'last_pred' in st.session_state:
            p = st.session_state['last_pred']
            st.metric("Predicted Stress Level", p['res'], f"{p['conf']:.1f}% Confidence")
            if p['res'] == "High":
                st.error("🚨 **High Stress Level Detected**")
                st.warning("Reach out: [UTS Counselor](https://sdsc.uts.edu.my/psychology-counselling/) | 03-7627 2929")
            st.success(p['advice'])
            
            if st.session_state['user_role'] != "Guest":
                st.write("Is this accurate?")
                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button("✅ Yes", use_container_width=True): 
                        supabase.table("user_feedback").insert({**p['inputs'], "student_id": st.session_state['username']}).execute()
                        st.toast("Verified!")
                with col_n:
                    if st.button("❌ No, correct it", use_container_width=True): st.session_state['feedback_mode'] = True

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
    res = supabase.table("user_history").select("*").eq("student_id", st.session_state['username']).order("created_at", desc=True).execute()
    if res.data:
        df_hist = pd.DataFrame(res.data)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else: st.info("No records found.")

elif page == "📝 UAT Survey Data":
    if st.session_state['user_role'] != "Admin": st.stop()
    st.title("📝 UAT Internal Feedback")
    res = supabase.table("user_feedback").select("*").execute()
    if res.data: st.dataframe(pd.DataFrame(res.data), use_container_width=True, hide_index=True)
    else: st.info("No feedback yet.")

elif page == "📊 Dashboard":
    if st.session_state['user_role'] != "Admin": st.stop()
    st.title("📊 UAT Real-Time Logs")
    res = supabase.table("user_history").select("*").order('created_at', desc=True).execute()
    if res.data:
        df_logs = pd.DataFrame(res.data)
        # 🌟 UI 修复：宽屏自适应 + 动态高度 🌟
        st.dataframe(
            df_logs, 
            use_container_width=True, 
            hide_index=True, 
            height=min(800, (len(df_logs) + 1) * 38)
        )
        csv = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download UAT CSV", csv, "uat_data.csv", "text/csv")
    else: st.info("No data yet.")

elif page == "📈 Data Analysis":
    if st.session_state['user_role'] != "Admin": st.stop()
    st.title("📈 Model Transparency")
    st.metric("Live Accuracy", f"{train_acc*100:.2f}%")
    if model_cm is not None: st.write("Confusion Matrix:", model_cm)

elif page == "⚙️ Account Settings":
    st.title("⚙️ Settings")
    st.write(f"ID: `{st.session_state['username']}` | Role: `{st.session_state['user_role']}`")
    with st.form("p_form"):
        new_p = st.text_input("New Password", type="password")
        if st.form_submit_button("Update"):
            supabase.table("users").update({"password_hash": make_hashes(new_p)}).eq("student_id", st.session_state['username']).execute()
            st.success("Updated!")
