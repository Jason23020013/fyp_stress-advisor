import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import time
import sqlite3
import hashlib  # 用于密码加密
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. 核心库导入与环境检查 ---
try:
    from supabase import create_client, Client
    import google.generativeai as genai
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    st.error(f"🚨 缺少必要的库: {e}. 请确保 requirements.txt 包含 supabase, google-generativeai, imbalanced-learn")
    st.stop()

# ==========================================
# 0. 权限与云端数据库配置 (新增)
# ==========================================

# 密码哈希处理函数
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# 从 Secrets 安全读取 Supabase 配置
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error("❌ 无法连接到 Supabase 云端数据库，请检查 Streamlit Secrets 配置。")
    st.stop()

# 初始化登录状态
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = "Student"
if 'feedback_mode' not in st.session_state:
    st.session_state['feedback_mode'] = False

# ==========================================
# 1. 登录与注册拦截界面 (新增)
# ==========================================
if not st.session_state['logged_in']:
    st.set_page_config(page_title="🔐 UTS Stress System Login", layout="centered")
    st.title("🎓 UTS 学生压力顾问系统")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["用户登录 (Login)", "新用户注册 (Register)"])
    
    with tab1:
        l_user = st.text_input("学号 (Student ID)", placeholder="例如: BCS23020013", key="login_u")
        l_pwd = st.text_input("密码 (Password)", type="password", key="login_p")
        if st.button("立即登录", use_container_width=True):
            # 查询云端数据库
            res = supabase.table("users").select("*").eq("student_id", l_user).execute()
            if res.data and check_hashes(l_pwd, res.data[0]['password_hash']):
                st.session_state['logged_in'] = True
                st.session_state['username'] = l_user
                st.session_state['user_role'] = res.data[0].get('role', 'Student')
                st.success(f"欢迎回来, {l_user}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("学号或密码错误，请重试。")
    
    with tab2:
        st.info("提示：注册后的默认身份为 Student。")
        r_user = st.text_input("设置学号 (Username)", placeholder="建议使用学号", key="reg_u")
        r_pwd = st.text_input("设置密码 (Password)", type="password", key="reg_p")
        if st.button("完成注册", use_container_width=True):
            if r_user and r_pwd:
                hashed = make_hashes(r_pwd)
                try:
                    supabase.table("users").insert({
                        "student_id": r_user, 
                        "password_hash": hashed, 
                        "role": "Student"
                    }).execute()
                    st.success("注册成功！现在请切换到‘登录’标签。")
                except:
                    st.error("注册失败：该学号可能已被注册。")
            else:
                st.error("学号和密码不能为空。")
    st.stop() # 未登录则停止运行后续代码

# ==========================================
# 2. 以下为您原本的所有代码逻辑 (完全保留)
# ==========================================

# --- DATABASE CONNECTION ---
def get_db_connection():
    conn = sqlite3.connect('student_stress.db', check_same_thread=False)
    return conn

# --- INTERNAL BRAIN (SQL Powered) ---
@st.cache_resource
def train_internal_model(force_retrain=False):
    if not os.path.exists('student_stress.db'):
        return None, None, 0, 0, None, None

    conn = get_db_connection()
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

    df['Lifestyle_Score'] = (df['Social_Hours_Per_Day'] + 
                             df['Physical_Activity_Hours_Per_Day'] + 
                             df['Extracurricular_Hours_Per_Day'])
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

# 侧边栏登出与欢迎词
st.sidebar.markdown(f"### 👋 欢迎, {st.session_state['username']}")
st.sidebar.write(f"当前身份: `{st.session_state['user_role']}`")
if st.sidebar.button("🚪 退出登录"):
    st.session_state['logged_in'] = False
    st.rerun()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3062/3062331.png", width=100)
st.sidebar.title("功能导航")

# 侧边栏菜单权限控制 (Admin 可见 Analysis 和 Dashboard)
menu_options = ["🏠 Home", "🤖 AI Predictor", "💬 AI Chatbot", "📝 User Survey"]
if st.session_state['user_role'] == "Admin":
    menu_options += ["📈 Data Analysis", "📊 Dashboard"]

page = st.sidebar.radio("前往页面", menu_options)

# --- API KEY SETUP ---
st.sidebar.markdown("---")
st.sidebar.header("🔑 AI 连接状态")
api_key = st.secrets["GEMINI_API_KEY"]

if api_key == "" or "PASTE_YOUR_KEY" in api_key:
    gemini_model = None
    st.sidebar.error("🔴 未发现 API Key")
else:
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
        st.sidebar.success("🟢 AI 已连接 (Gemini 2.5)")
    except Exception as e:
        gemini_model = None
        st.sidebar.error(f"🔴 连接错误: {e}")

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
            st.write("Is this result correct?")
            cy, cn = st.columns(2)
            if cy.button("✅ Yes, Correct"):
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("INSERT INTO user_feedback (Study_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, Physical_Activity_Hours_Per_Day, Extracurricular_Hours_Per_Day, GPA, Stress_Level) VALUES (?, ?, ?, ?, ?, ?, ?)", p['inputs'])
                conn.commit(); conn.close()
                st.success("Feedback saved!")
            if cn.button("❌ No, It's Wrong"):
                st.session_state['feedback_mode'] = True

            if st.session_state.get('feedback_mode', False):
                with st.form("correction_form"):
                    correct_label = st.selectbox("Correct Stress Level", ["Low", "Moderate", "High"])
                    if st.form_submit_button("Submit Correction"):
                        conn = get_db_connection(); cur = conn.cursor()
                        inputs = list(p['inputs']); inputs[-1] = correct_label 
                        cur.execute("INSERT INTO user_feedback (Study_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, Physical_Activity_Hours_Per_Day, Extracurricular_Hours_Per_Day, GPA, Stress_Level) VALUES (?, ?, ?, ?, ?, ?, ?)", tuple(inputs))
                        conn.commit(); conn.close()
                        st.success("Correction Saved!"); st.session_state['feedback_mode'] = False; st.rerun()

# --- PAGE: CHATBOT ---
elif page == "💬 AI Chatbot":
    st.title("💬 Gemini Student Counselor")
    if not gemini_model:
        st.warning("⚠️ Gemini is disconnected.")
    else:
        if "messages" not in st.session_state: st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])
        if prompt := st.chat_input("How are you feeling today?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                system_prompt = "You are an empathetic AI Student Counselor. Support the student warmly and concisely."
                response = gemini_model.generate_content(system_prompt + history_text)
                message_placeholder.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

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
                conn = get_db_connection(); cur = conn.cursor()
                cur.execute("INSERT INTO user_feedback (Study_Hours_Per_Day, Sleep_Hours_Per_Day, Social_Hours_Per_Day, Physical_Activity_Hours_Per_Day, Extracurricular_Hours_Per_Day, GPA, Stress_Level) VALUES (?, ?, ?, ?, ?, ?, ?)", (s_study, s_sleep, s_social, s_phys, s_extra, s_gpa, s_stress))
                conn.commit(); conn.close(); st.success("Data added.")

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
        except: st.write("No data found.")
        finally: conn.close()
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
    finally: conn.close()
    col1, col2 = st.columns(2)
    col1.metric("Testing Accuracy", f"{test_acc*100:.1f}%")
    col2.metric("New Feedback", count_feed)
    st.markdown("---")
    st.subheader("⚙️ Model Maintenance")
    if st.button("🔄 Retrain Model"):
        if count_feed == 0: st.warning("No new data.")
        else:
            with st.spinner("Retraining..."):
                st.cache_resource.clear()
                model, le, tr_acc, te_acc, cm, feats = train_internal_model(force_retrain=True)
                st.success("Retrained Successfully!")
