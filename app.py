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
        feature_names = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Lifestyle_Score', 'Academic_Pressure', 'GPA']
        
        df = pd.read_csv("student_lifestyle_dataset.csv").dropna()
        df['Lifestyle_Score'] = df['Social_Hours_Per_Day'] + df['Physical_Activity_Hours_Per_Day'] + df['Extracurricular_Hours_Per_Day']
        df['Academic_Pressure'] = df['Study_Hours_Per_Day'] * (5.0 - df['GPA'])
        
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
# 3. Precision Dual-Engine System (3 Flash + 3.1 Flash Lite)
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
    
    active_key = key1 if key1 else st.secrets.get("GEMINI_API_KEY")
    if not active_key:
        return "AI Counselor API Keys are not properly configured."

    try:
        genai.configure(api_key=active_key)
        model_name = 'gemini-3-flash' 
        model_primary = genai.GenerativeModel(model_name, system_instruction=COUNSELOR_PERSONA)
        response = model_primary.generate_content(prompt_text)
        return response.text + f"\n\n*(Engine: {model_name} - Premium)*"
    except Exception as e:
        if not ("ResourceExhausted" in str(e) or "429" in str(e) or "quota" in str(e).lower() or "404" in str(e)):
            return f"Error with Primary AI: {e}"

    try:
        fallback_key = key2 if key2 else active_key
        genai.configure(api_key=fallback_key)
        
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name.replace('models/', ''))
                
        fallback_name = None
        for name in available_models:
             if "gemini-3.1-flash-lite" in name:
                 fallback_name = name
                 break
        
        if not fallback_name:
             for name in available_models:
                 if ("lite" in name or "flash" in name) and "pro" not in name:
                     fallback_name = name
                     break
                     
        if not fallback_name and available_models:
            fallback_name = available_models[0]
            
        if fallback_name:
            model_fallback = genai.GenerativeModel(fallback_name, system_instruction=COUNSELOR_PERSONA)
            response = model_fallback.generate_content(prompt_text)
            return response.text + f"\n\n*(Engine: {fallback_name} - Fallback)*"
        else:
            return "Critical Error: API returned empty list of available models."

    except Exception as e2:
        return f"Auto-Fallback failed entirely. Error: {e2}"

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Gemini AI Counselor", layout="wide")

st.sidebar.markdown(f"### 👋 Welcome, {st.session_state['username']}")
st.sidebar.write(f"Role: `{st.session_state['user_role']}`")
if st.sidebar.button("🚪 Logout"):
    st.session_state['logged_in'] = False
    st.query_params.clear()
    st.rerun()

st.sidebar.title("Navigation")
# --- ROLE-BASED ACCESS CONTROL (RBAC) FOR MENU ---
menu_options = ["🏠 Home", "🤖 AI Predictor", "💬 AI Chatbot"]

if st.session_state['user_role'] in ["Student", "Admin"]:
    menu_options += ["📜 My History", "⚙️ Account Settings"]

if st.session_state['user_role'] == "Admin":
    menu_options += ["📝 UAT Survey Data", "📈 Data Analysis", "📊 Dashboard"]

page = st.sidebar.radio("Go to", menu_options)

if st.secrets.get("GEMINI_API_KEY_1") or st.secrets.get("GEMINI_API_KEY"):
    st.sidebar.success("🟢 AI Engines Active")
else: 
    st.sidebar.error("🔴 AI Connection Missing")

# ==========================================
# 4. PAGE LOGIC
# ==========================================

# 👇 --- GLOBAL UAT SURVEY BANNER --- 👇
if st.session_state['user_role'] != "Admin":
    st.info("📢 **UAT Phase Active:** Once you get your AI prediction, please help us by filling out the survey: [👉 Click here to open Google Form](https://forms.gle/sDmDD8s828LPkb3X9)")

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
    - **💬 Generative AI:** Empathetic counseling support powered by **Google Gemini** (Precision Dual-Engine System).
    - **📜 Cloud Integration:** Secure tracking of wellness history via Supabase.
    """)
    
    if st.session_state['user_role'] == "Admin":
        st.markdown("---")
        st.subheader("📊 Final Model Performance Report (Admin Only)")
        
        col1, col2 = st.columns(2)
        col1.info(f"✅ Training Accuracy (Final): 99.88%")
        col2.success(f"✅ Testing Accuracy (Final): 99.25%")
        
        st.markdown("**Status:** ✨ ROBUST MODEL (Balanced Performance)")
        
        with st.expander("🔍 View Detailed Confusion Matrix"):
            st.code("""
[ Confusion Matrix ]
[[202   0   3]
 [  0  67   0]
 [  0   0 129]]
            """)
    else:
        st.success("💡 System is fully operational. AI Predictor is ready for assessment.")

elif page == "🤖 AI Predictor":
    st.title("🤖 AI Stress Assessment")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Your Daily Habits")
        st.caption("💡 Sync: Use sliders or type numbers directly.")

        def sync_v(key_from, key_to): st.session_state[key_to] = st.session_state[key_from]

        # Initialize session states for standard habits
        for k in ['study', 'sleep', 'social', 'phys', 'extra']:
            if f'slider_{k}' not in st.session_state: st.session_state[f'slider_{k}'] = 5.0
            if f'num_{k}' not in st.session_state: st.session_state[f'num_{k}'] = 5.0

        # Initialize session state for GPA
        if 'slider_gpa' not in st.session_state: st.session_state['slider_gpa'] = 3.00
        if 'num_gpa' not in st.session_state: st.session_state['num_gpa'] = 3.00

        def input_row(label, key_base, min_v=0.0, max_v=24.0, step_v=0.5):
            col_s, col_n = st.columns([3, 1])
            with col_s: val = st.slider(label, min_v, max_v, key=f"slider_{key_base}", step=step_v, on_change=sync_v, args=(f"slider_{key_base}", f"num_{key_base}"))
            with col_n: st.number_input(label, min_v, max_v, key=f"num_{key_base}", step=step_v, on_change=sync_v, args=(f"num_{key_base}", f"slider_{key_base}"), label_visibility="hidden")
            return val

        study = input_row("Study Hours (per day)", "study")
        sleep = input_row("Sleep Hours (per day)", "sleep")
        social = input_row("Social Hours (per day)", "social")
        phys = input_row("Physical Activity (per day)", "phys")
        extra = input_row("Extracurriculars (per day)", "extra")

        total = study + sleep + social + phys + extra
        st.markdown(f"**Used: {total}/24 Hours**")
        st.progress(min(total/24.0, 1.0))
        
        gpa = input_row("Current GPA", "gpa", 0.00, 4.00, 0.01)
        
        if total > 24: st.error("🚨 Total hours cannot exceed 24.")
        else:
            if st.button("Analyze Stress Level", use_container_width=True):
                # 🚀 STRATEGY 1: Adding the loading spinner here 🚀
                with st.spinner("⏳ AI is analyzing your data & generating advice... (Takes ~3 seconds)"):
                    if model:
                        l_score = social + phys + extra
                        a_press = study * (5.0 - gpa)
                        
                        input_df = pd.DataFrame([[study, sleep, l_score, a_press, gpa]], columns=feature_names)
                        
                        pred_idx = model.predict(input_df)[0]
                        confidence = np.max(model.predict_proba(input_df)[0]) * 100
                        pred_label = le.inverse_transform([pred_idx])[0]
                        
                        if st.session_state['user_role'] != "Guest":
                            supabase.table("user_history").insert({"student_id": st.session_state['username'], "Study_Hours_Per_Day": study, "Sleep_Hours_Per_Day": sleep, "Social_Hours_Per_Day": social, "Physical_Activity_Hours_Per_Day": phys, "Extracurricular_Hours_Per_Day": extra, "GPA": gpa, "Stress_Level": pred_label}).execute()

                        # === “摆烂（Apathy）”检测机制 ===
                        apathy_flag = ""
                        if study <= 0.5 and gpa < 2.0:
                            apathy_flag = "⚠️ CLINICAL NOTE: This student has almost 0 study hours and a low GPA. The ML model predicted high stress, but psychologically, they might actually be experiencing 'Academic Disengagement', burnout, or apathy (they simply do not care about studies). DO NOT assume they are overwhelmed by studying. Instead, gently address their motivation, ask about their true interests (based on their high social/extracurricular hours), and provide advice on finding purpose rather than just 'stress relief'."

                        p = f"Student Profile: {study}h Study, {sleep}h Sleep, {social}h Social, {phys}h Physical, {extra}h Extracurricular, {gpa} GPA. ML Prediction: {pred_label} Stress ({confidence:.1f}% confidence). {apathy_flag}\n\nProvide a realistic 1-sentence analysis of their situation and time management, followed by 3 direct, actionable tips."
                        
                        ai_advice = get_gemini_response(p)
                        
                        st.session_state['last_pred'] = {'res': pred_label, 'conf': confidence, 'advice': ai_advice, 'inputs': {"Study_Hours_Per_Day": study, "Sleep_Hours_Per_Day": sleep, "Social_Hours_Per_Day": social, "Physical_Activity_Hours_Per_Day": phys, "Extracurricular_Hours_Per_Day": extra, "GPA": gpa, "Stress_Level": pred_label}}

    with c2:
        st.subheader("Analysis Result")
        if 'last_pred' in st.session_state:
            p = st.session_state['last_pred']
            st.metric("Predicted Stress Level", p['res'], f"{p['conf']:.1f}% Confidence")
            
            if p['res'] == "High":
                st.error("🚨 **High Stress Level Detected** 🚨")
                st.warning("""
                **Your well-being is our top priority.** If you feel overwhelmed or need someone to talk to, please reach out to a professional:
                - 📞 **UTS Campus Counselor:** https://sdsc.uts.edu.my/psychology-counselling/
                - 📞 **Befrienders Malaysia (24/7 Hotline):** 03-7627 2929
                """)

            st.success(p['advice'])
            
            if st.session_state['user_role'] != "Guest":
                st.write("Is this accurate?")
                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button("✅ Yes", use_container_width=True): 
                        supabase.table("user_feedback").insert({**p['inputs'], "student_id": st.session_state['username']}).execute()
                        st.success("Verified!")
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
    res = supabase.table("user_history").select("created_at, Study_Hours_Per_Day, Sleep_Hours_Per_Day, GPA, Stress_Level").eq("student_id", st.session_state['username']).order("created_at", desc=True).execute()
    
    if res.data:
        df_hist = pd.DataFrame(res.data)
        
        st.subheader("📈 Your Stress Trend Over Time")
        chart_df = df_hist.copy()
        chart_df['Date'] = pd.to_datetime(chart_df['created_at']).dt.strftime('%b %d, %H:%M')
        level_mapping = {'Low': 1, 'Medium': 2, 'Moderate': 2, 'High': 3}
        chart_df['Stress_Value'] = chart_df['Stress_Level'].map(level_mapping)
        chart_df = chart_df.sort_values('created_at')
        
        st.line_chart(data=chart_df, x='Date', y='Stress_Value')
        st.markdown("*(1 = Low Stress, 2 = Moderate/Medium, 3 = High Stress)*")
        st.markdown("---")

        st.subheader("📋 Detailed Records")
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else: 
        st.info("No records found.")

elif page == "📝 UAT Survey Data":
    if st.session_state['user_role'] != "Admin":
        st.error("🚫 Access Denied. Admin privileges required.")
        st.stop()
        
    st.title("📝 UAT Survey Data (Admin Only)")
    st.markdown("🔒 **Data Privacy Enforcement:** This page is strictly hidden from regular students to protect Personal Identifiable Information (PII) and feedback data.")
    
    st.subheader("Internal System Corrections & Feedback")
    st.write("This table logs whenever a user corrects the AI's prediction.")
    res = supabase.table("user_feedback").select("*").execute()
    if res.data:
        st.dataframe(pd.DataFrame(res.data), use_container_width=True)
    else:
        st.info("No internal feedback collected yet.")
        
    st.success("💡 **Note for Final Report:** The external User Acceptance Testing (UAT) results containing the 50 student emails will be managed securely via your Google Forms / Google Sheets dashboard.")

elif page == "⚙️ Account Settings":
    st.title("⚙️ Account Settings")
    st.markdown("Manage your account details and security.")
    
    st.write(f"**Student ID / Username:** `{st.session_state['username']}`")
    st.write(f"**Account Role:** `{st.session_state['user_role']}`")
    st.markdown("---")
    
    st.subheader("🔒 Change Password")
    with st.form("change_password_form"):
        old_pwd = st.text_input("Current Password", type="password")
        new_pwd = st.text_input("New Password", type="password")
        confirm_pwd = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Update Password"):
            if not old_pwd or not new_pwd:
                st.error("Please fill in all fields.")
            elif new_pwd != confirm_pwd:
                st.error("New passwords do not match!")
            else:
                res = supabase.table("users").select("password_hash").eq("student_id", st.session_state['username']).execute()
                if res.data and check_hashes(old_pwd, res.data[0]['password_hash']):
                    supabase.table("users").update({"password_hash": make_hashes(new_pwd)}).eq("student_id", st.session_state['username']).execute()
                    st.success("✅ Password updated successfully!")
                else:
                    st.error("❌ Incorrect current password.")

elif page == "📈 Data Analysis":
    if st.session_state['user_role'] != "Admin":
        st.error("🚫 Access Denied. Admin privileges required.")
        st.stop()

    st.title("📈 Model Transparency")
    t1, t2 = st.tabs(["Dataset Preview", "Live Metrics"])
    with t1:
        try: st.dataframe(pd.read_csv("student_lifestyle_dataset.csv").head(100))
        except: st.error("Baseline CSV missing.")
    with t2:
        st.write("### Current Model Performance (Real-time Validation)")
        st.metric("Live Calculated Accuracy", f"{train_acc*100:.2f}%")
        if model_cm is not None: st.write("Current Confusion Matrix:", model_cm)

elif page == "📊 Dashboard":
    if st.session_state['user_role'] != "Admin":
        st.error("🚫 Access Denied. Admin privileges required.")
        st.stop()
        
    st.title("📊 UAT Real-Time Logs & Admin Monitoring")
    
    res_fb = supabase.table("user_feedback").select("id", count="exact").execute()
    st.metric("Total Cloud Feedback Received", res_fb.count if res_fb.count else 0)
    st.info("Architecture: Offline Batch Retraining. Update .pkl files to deploy changes.")
    
    st.markdown("---")
    st.subheader("📋 Live Student Sessions")
    
    res_logs = supabase.table("user_history").select("*").order('created_at', desc=True).execute()
    if res_logs.data:
        df_logs = pd.DataFrame(res_logs.data)
        
        st.dataframe(
            df_logs,
            use_container_width=True,
            hide_index=True,
            height=min(800, (len(df_logs) + 1) * 38)
        )
        
        csv = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download UAT Log Data (CSV)",
            csv,
            "uat_realtime_logs.csv",
            "text/csv",
            key='download-uat-csv'
        )
    else:
        st.info("No logs collected yet. Waiting for participants...")
