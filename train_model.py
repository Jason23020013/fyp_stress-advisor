import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

def train_offline():
    print("🚀 Starting Offline Training System...")
    
    # 1. Connect to Database
    db_path = 'student_stress.db'
    if not os.path.exists(db_path):
        print(f"❌ Error: Database '{db_path}' not found. Please run 'init_db.py' first.")
        return

    print("🔌 Connecting to SQLite Database...")
    conn = sqlite3.connect(db_path)

    # 2. Fetch Data
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
        print(f"✅ Loaded {len(df)} records from SQL Database.")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return
    finally:
        conn.close()

    # 3. Feature Engineering
    print("⚙️ Engineering Features...")
    df['Lifestyle_Score'] = (df['Social_Hours_Per_Day'] + 
                             df['Physical_Activity_Hours_Per_Day'] + 
                             df['Extracurricular_Hours_Per_Day'])
    
    df['Academic_Pressure'] = df['GPA'] * df['Study_Hours_Per_Day']

    features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Lifestyle_Score', 'Academic_Pressure', 'GPA']
    X = df[features]
    y = df['Stress_Level']

    # 4. Encoding & Splitting
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 5. SMOTE
    print("⚖️ Balancing dataset with SMOTE...")
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    # 6. Train Model
    print("🧠 Training Random Forest (max_depth=2)...")
    model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=2,           
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_bal, y_bal)

    # ==========================================
    # 7. EVALUATE & SHOW METRICS TOGETHER
    # ==========================================
    
    # Calculate Accuracies
    train_pred = model.predict(X_bal)
    train_acc = accuracy_score(y_bal, train_pred)

    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print("\n" + "="*50)
    print("📊 FINAL MODEL PERFORMANCE REPORT")
    print("="*50)
    
    # 1. Accuracy Section
    print(f"✅ Training Accuracy:  {train_acc*100:.2f}%")
    print(f"✅ Testing Accuracy:   {test_acc*100:.2f}%")
    gap = train_acc - test_acc
    if gap > 0.10:
        print(f"⚠️  Status: OVERFITTING DETECTED (Gap: {gap*100:.1f}%)")
    else:
        print(f"✨  Status: ROBUST MODEL (Balanced Performance)")
    
    print("-" * 50)
    
    # 2. Detailed Metrics Section (Kept Together)
    print("🔍 DETAILED METRICS")
    
    print("\n[ Confusion Matrix ]")
    print(confusion_matrix(y_test, test_pred))
    
    

    print("\n[ Classification Report ]")
    print(classification_report(y_test, test_pred, target_names=le.classes_))
    
    print("="*50 + "\n")

    # 8. Save
    print("💾 Saving Model & Encoder...")
    joblib.dump(model, 'student_stress_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("✅ Model Saved Successfully.")

if __name__ == "__main__":
    train_offline()